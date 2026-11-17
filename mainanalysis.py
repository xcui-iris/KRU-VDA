import os
import re
import pathlib
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from scipy.stats import gaussian_kde
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")


# ============================================================================
# Setup: paths and parameters
# ============================================================================

REPO = pathlib.Path(__file__).resolve().parent
RAW = pathlib.Path(os.environ.get("KRU_RAW_DATA", REPO / "data" / "raw"))
VDA_CSV = REPO / "data" / "vda_monthly.csv"
OUT = REPO / "output"
TAB, FIG = OUT / "tables", OUT / "figures"

AEB_FILE = RAW / "historical aeb data - malone.xlsx"
WORD_CLOUD_DIR = RAW / "Word Cloud Data"
EQUIP_SHIPMENTS = RAW / "equipment_price_index" / "monthly.csv"
EQUIP_INVENTORIES = RAW / "equipment_price_index" / "monthly,_end_of_period.csv"
COT_DIR = RAW / "cot"

SAMPLE_START, SAMPLE_END = pd.Timestamp("2015-10-01"), pd.Timestamp("2025-09-01")
MONTHS = pd.date_range(SAMPLE_START, SAMPLE_END, freq="MS")
BASELINE = ("2016-10-01", "2019-12-01")
WINDOW = 12
SEED = 42

LDA_K, LDA_SEED = 7, 13
LDA_MIN_DF, LDA_MAX_DF = 10, 0.5
LDA_DOC_PRIOR, LDA_WORD_PRIOR, LDA_MAX_ITER = 0.1, 0.01, 40
LDA_TRAIN_SHARE, LDA_SPLIT_SEED = 0.8, 2026

MAX_P = 4
SETS = {"composites": ["V", "D", "A"], "quantitative": ["V_q", "D_q", "A_q"], "text": ["V_t", "D_t", "A_t"]}
OUTCOMES = ["equipment_shipments", "equipment_inventories", "cot_net_commercial", "cot_hedge_turnover"]
HALVES = [f"{k}_{h}" for k in "VDA" for h in "qt"]

EVENT_ONSETS = [("2016 Election", "2016-11"), ("Tariffs Begin", "2018-03"), ("Trade War Escalation", "2019-05"),
                ("COVID-19 Onset", "2020-03"), ("Fed Rate Hikes", "2022-03")]
EVENT_LINES = {"2016 election": "2016-11-01", "US-China\ntariffs": "2018-03-01", "Trade-war\nescalation": "2019-05-01",
               "Phase One\ndeal": "2019-12-01", "COVID-19": "2020-03-01", "Fed rate\nhikes": "2022-03-01"}
COLORS = {"V": "#E69F00", "D": "#56B4E9", "A": "#CC79A7", "AEB": "#333333"}
NAMES = {"V": "Volatility", "D": "Dispersion", "A": "Ambiguity"}


def zscore(s, base=BASELINE):
    b = s.loc[base[0]:base[1]].dropna()
    return (s - b.mean()) / b.std()


def roll(s, w=WINDOW):
    return s.rolling(w, min_periods=w).mean()


def raw_data_available():
    return AEB_FILE.exists() and WORD_CLOUD_DIR.exists()


# ============================================================================
# Step 1. Load raw data
# ============================================================================

def load_aeb_individual():
    df = pd.read_excel(AEB_FILE, sheet_name="Data", header=None).rename(columns={0: "Month", 1: "Year", 19: "AEB"})
    for c in ["Month", "Year", "AEB"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Year", "Month", "AEB"])
    df["Year"] = df["Year"].astype(int).apply(lambda y: y + 2000 if y < 100 else y)
    df["Date"] = pd.to_datetime(df.Year.astype(str) + "-" + df.Month.astype(int).astype(str).str.zfill(2) + "-01")
    return df[(df.Date >= SAMPLE_START) & (df.Date <= SAMPLE_END)]


def load_answers():
    rows = []
    for f in sorted(WORD_CLOUD_DIR.rglob("word cloud - *.xlsx")):
        yy, mm = f.stem.split(" - ")[-1].split("-")
        d = pd.Timestamp(year=2000 + int(yy), month=int(mm), day=1)
        if not SAMPLE_START <= d <= SAMPLE_END:
            continue
        xl = pd.ExcelFile(f)
        sheet = "Comments" if "Comments" in xl.sheet_names else xl.sheet_names[0]
        texts = xl.parse(sheet, header=None)[1].dropna().astype(str).tolist()
        rows += [dict(Date=d, resp_id=f"{d:%Y%m}_{j:04d}", raw=s.strip()) for j, s in enumerate(texts)]
    return pd.DataFrame(rows).sort_values(["Date", "resp_id"]).reset_index(drop=True)


def load_outcomes():
    def fred(path, col):
        s = pd.read_csv(path, parse_dates=["observation_date"], index_col="observation_date", na_values=[".", ""])[col]
        s = pd.to_numeric(s, errors="coerce").dropna()
        s.index = s.index.to_period("M").to_timestamp()
        return s

    def seasonal_adjust(s):
        s = s.dropna()
        return np.exp(np.log(s) - STL(np.log(s), period=12, robust=True).fit().seasonal)

    cols = ["Report_Date_as_MM_DD_YYYY", "CFTC_Commodity_Code", "Market_and_Exchange_Names", "Comm_Positions_Long_All",
            "Comm_Positions_Short_All", "Change_in_Comm_Long_All", "Change_in_Comm_Short_All"]
    frames = []
    for d in sorted(COT_DIR.glob("dea_fut_xls_*")):
        if (d / "annual.xls").exists():
            df = pd.read_excel(d / "annual.xls", sheet_name="XLS", engine="xlrd", usecols=cols)
            name = df["Market_and_Exchange_Names"].str.upper()
            keep = ((df["CFTC_Commodity_Code"] == 2) & name.str.contains("CORN")) | \
                   ((df["CFTC_Commodity_Code"] == 5) & name.str.contains("SOYBEANS") & ~name.str.contains("MINI"))
            frames.append(df[keep])
    w = pd.concat(frames)
    w["date"] = pd.to_datetime(w["Report_Date_as_MM_DD_YYYY"], errors="coerce")
    for c in cols[3:]:
        w[c] = pd.to_numeric(w[c], errors="coerce")
    w = w.dropna(subset=["date"]).drop_duplicates(["date", "CFTC_Commodity_Code"]).sort_values("date")

    net = w.assign(net=w.Comm_Positions_Long_All - w.Comm_Positions_Short_All) \
        .groupby(["CFTC_Commodity_Code", pd.Grouper(key="date", freq="ME")]).net.last().unstack(0).sum(axis=1, min_count=1)
    net.index = net.index.to_period("M").to_timestamp()
    w["turn"] = w.Change_in_Comm_Long_All.abs() + w.Change_in_Comm_Short_All.abs()
    g = w.groupby(pd.Grouper(key="date", freq="ME"))
    turn = (g.turn.sum(min_count=1) / g.date.nunique()).dropna()
    turn.index = turn.index.to_period("M").to_timestamp()

    return pd.concat({"equipment_shipments": fred(EQUIP_SHIPMENTS, "A33AVS"),
                      "equipment_inventories": seasonal_adjust(fred(EQUIP_INVENTORIES, "U33ATI")),
                      "cot_net_commercial": net,
                      "cot_hedge_turnover": seasonal_adjust(turn)}, axis=1).reindex(MONTHS)


# ============================================================================
# Step 2. Classify open-ended answers
# ============================================================================

NONRESPONSE = re.compile(
    r"^\W*(?:"
    r"(?:no|np|nol)\W*(?:(?!restrict|rest\b|research|resource|respect|repeal|represent|republic)"
    r"re[sp]\w*|comments?|answers?)(?:\W+(?:given|recorded|provided))?"
    r"|noresponse|none given|refused|refuse|declined?|blank|skip(?:ped)?|n/?a"
    r")?\W*$", re.IGNORECASE)


def substantive_answers(answers):
    answers = answers.copy()
    answers["substantive"] = ~answers.raw.apply(lambda s: (not s) or bool(NONRESPONSE.match(s)))
    answers["words"] = np.where(answers.substantive, answers.raw.str.count(r"[A-Za-z']+"), 0)
    sub = answers[answers.substantive].copy()
    g = sub.groupby("Date")
    length = pd.DataFrame({"mean_logwords": g.words.apply(lambda w: np.log1p(w).mean()),
                           "share_1word": g.words.apply(lambda w: (w == 1).mean())}).reindex(MONTHS)
    return sub, length


# ============================================================================
# Step 3. Quantitative sub-measures: V_q, D_q, A_q
# ============================================================================

GRID = np.linspace(-5, 5, 200)


def density(x):
    z = (x - x.mean()) / x.std(ddof=1)
    d = np.clip(gaussian_kde(z, bw_method="silverman")(GRID), 1e-12, None)
    return d / d.sum()


def kl(p, q):
    return float(np.sum(p * np.log(p / q)))


def quantitative_measures(ind):
    aeb_mean = ind.groupby("Date").AEB.mean().reindex(MONTHS)
    rows, prev = [], None
    for d, g in ind.groupby("Date"):
        x = g.AEB.dropna().values
        dn = density(x)
        rows.append({"Date": d, "D_sd_m": x.std(ddof=1), "A_kl_m": kl(dn, prev) if prev is not None else np.nan})
        prev = dn
    b = pd.DataFrame(rows).set_index("Date").reindex(MONTHS)
    return pd.DataFrame({"AEB_mean": aeb_mean,
                         "V_q": aeb_mean.diff().rolling(WINDOW, min_periods=WINDOW).std(),
                         "D_q": b.D_sd_m,
                         "A_kl_m": b.A_kl_m,
                         "A_q": roll(b.A_kl_m)}, index=MONTHS)


# ============================================================================
# Step 4. Text cleaning and topic model
# ============================================================================

CONTRACTIONS = {r"\bwon't\b": "will not", r"\bcan't\b": "can not", r"\bain't\b": "is not", r"n't\b": " not",
                r"'re\b": " are", r"'ve\b": " have", r"'ll\b": " will", r"'d\b": " would", r"'m\b": " am", r"'s\b": ""}
DOMAIN_STOP = {
    "concern", "concerned", "concerning", "worry", "worried", "issue", "affect", "operation", "future", "near",
    "farmer", "farm", "farming", "need", "going", "get", "getting", "would", "like", "make", "think", "lot", "really",
    "much", "many", "thing", "also", "well", "want", "see", "keep", "know", "one", "way", "time", "year", "right",
    "could", "still", "even", "every", "come", "people", "say", "said", "says", "feel", "feels", "seem", "go", "put",
    "take", "able", "trying", "try", "hope", "dont", "doesnt",
}


def stop_words():
    from nltk.corpus import stopwords
    return set(stopwords.words("english")) | DOMAIN_STOP


def lemma(w):
    if len(w) > 4 and w.endswith(("xes", "ches", "shes", "sses", "zes")):
        return w[:-2]
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is", "ous")):
        return w[:-1]
    return w


def clean(raw, sw):
    s = raw.lower().replace("’", "'")
    for p, r in CONTRACTIONS.items():
        s = re.sub(p, r, s)
    toks = [lemma(w) for w in re.sub(r"[^a-z\s]", " ", s).split()]
    return [w for w in toks if len(w) > 2 and w not in sw]


def document_term(raw_answers, sw):
    from sklearn.feature_extraction.text import CountVectorizer
    docs = [" ".join(clean(r, sw)) for r in raw_answers]
    vec = CountVectorizer(min_df=LDA_MIN_DF, max_df=LDA_MAX_DF, token_pattern=r"\S+")
    return vec.fit_transform(docs).tocsr(), np.array(vec.get_feature_names_out())


def fit_lda(dtm, k, seed):
    from sklearn.decomposition import LatentDirichletAllocation
    usable = np.asarray(dtm.sum(axis=1)).ravel() > 0
    train = np.random.default_rng(LDA_SPLIT_SEED).random(dtm.shape[0]) < LDA_TRAIN_SHARE
    lda = LatentDirichletAllocation(n_components=k, learning_method="batch", max_iter=LDA_MAX_ITER, random_state=seed,
                                    doc_topic_prior=LDA_DOC_PRIOR, topic_word_prior=LDA_WORD_PRIOR,
                                    n_jobs=-1).fit(dtm[train & usable])
    return lda, lda.transform(dtm), usable


def mean_pairwise_jsd(p):
    a, b = p[:, None, :], p[None, :, :]
    m = (a + b) / 2
    kl_ = lambda x, y: np.sum(np.where(x > 0, x * np.log(np.where(x > 0, x, 1) / y), 0), axis=2)
    j = 0.5 * kl_(np.broadcast_to(a, m.shape), m) + 0.5 * kl_(np.broadcast_to(b, m.shape), m)
    return j[np.triu_indices(len(p), 1)].mean()


def monthly_distance(theta, usable, dates):
    return pd.Series({pd.Timestamp(m): mean_pairwise_jsd(theta[(dates == m) & usable])
                      for m in np.unique(dates)}).reindex(MONTHS)


def residualize(y, x):
    d = pd.concat([y.rename("y"), x], axis=1).dropna()
    fit = sm.OLS(d.y, sm.add_constant(d[x.columns])).fit()
    return (fit.resid + d.y.mean()).reindex(y.index)


# ============================================================================
# Step 5. Text sub-measures: V_t, D_t, A_t
# ============================================================================

DONT_KNOW = r"(?:don'?t|do not)\s+know\b"
AMBIGUITY_PHRASES = {
    "explicit hedging": [
        r"\bhard to tell\b",
        r"\b(?:it|all|everything|that|a lot)\s+depends\b",
        r"(?<!which )\bdepends(?:\s+(?:on|upon))?\s+(?:who|what|how|whether|if|the election|election|the outcome|the results|the president)\b",
        r"(?:(?:^|\|)\s*|\b(?:i|we|he|she)\s+(?:really\s+|just\s+|still\s+)?)" + DONT_KNOW],
    "competing forces": [r"\bup and down\b", r"\bback and forth\b", r"\bboth sides\b", r"\bgood and bad\b", r"\beither way\b"],
    "directional agnosticism": [r"\bwhich way\b", r"\bwho knows\b", r"\bwait and see\b", r"\bhard to predict\b",
                                r"\bhard to know\b", r"\bmixed\b"],
}
AMBIG = {grp: re.compile("|".join(f"(?:{p})" for p in pats)) for grp, pats in AMBIGUITY_PHRASES.items()}
UNRESTRICTED = {"depends": r"\bdepends\b", "don't know": r"\b" + DONT_KNOW}


def ambiguity_norm(raw):
    s = raw.str.lower().str.replace("’", "'").str.replace(r"[.,;:!?]+", " | ", regex=True)
    return s.str.replace(r"[^a-z'| ]", " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()


def ambiguity_group(text):
    return next((grp for grp, rx in AMBIG.items() if rx.search(text)), "")


def concern_churn(sub, sw):
    toks = pd.Series([clean(r, sw) for r in sub.raw], index=sub.index)
    top = {pd.Timestamp(m): Counter(w for t in toks[sub.Date == m] for w in t).most_common(20)
           for m in np.unique(sub.Date.values)}
    words = {m: {w for w, _ in c} for m, c in top.items()}
    keys = [m for m in MONTHS if m in words]
    jac = pd.Series({b: 1 - len(words[a] & words[b]) / len(words[a] | words[b])
                     for a, b in zip(keys, keys[1:])}).reindex(MONTHS)
    return jac, top


def topic_distance(sub, length, dtm, k, seed):
    lda, theta, usable = fit_lda(dtm, k, seed)
    pj = monthly_distance(theta, usable, sub.Date.values)
    return residualize(pj, length[["mean_logwords", "share_1word"]]), lda, theta, usable


def ambiguity_share(sub):
    hedge = ambiguity_norm(sub.raw).apply(ambiguity_group) != ""
    return hedge.groupby(sub.Date).mean().reindex(MONTHS)


def text_measures(sub, length):
    sw = stop_words()
    dtm, _ = document_term(sub.raw.tolist(), sw)
    jac, _ = concern_churn(sub, sw)
    a_t_m = ambiguity_share(sub)
    return pd.DataFrame({"V_t_m": jac,
                         "V_t": jac.rolling(WINDOW, min_periods=WINDOW).std(),
                         "D_t": topic_distance(sub, length, dtm, LDA_K, LDA_SEED)[0],
                         "A_t_m": a_t_m,
                         "A_t": roll(a_t_m)}, index=MONTHS)


# ============================================================================
# Step 6. Standardize and combine into V, D, A
# ============================================================================

def combine(q, t):
    v = pd.DataFrame({"V_q": zscore(q.V_q), "V_t": zscore(t.V_t), "D_q": zscore(q.D_q), "D_t": zscore(t.D_t),
                      "A_q": zscore(q.A_q), "A_t": zscore(t.A_t)}, index=MONTHS)
    for k in "VDA":
        v[k] = v[[f"{k}_q", f"{k}_t"]].mean(axis=1, skipna=False)
    v["AEB_mean"] = q.AEB_mean
    return v[["AEB_mean", "V", "D", "A"] + HALVES]


def build_monthly_series():
    ind = load_aeb_individual()
    sub, length = substantive_answers(load_answers())
    q = quantitative_measures(ind)
    t = text_measures(sub, length)
    v = combine(q, t).join(load_outcomes())
    v.index.name = "date"
    v.to_csv(VDA_CSV)
    return v


def load_monthly_series():
    return pd.read_csv(VDA_CSV, parse_dates=["date"], index_col="date").reindex(MONTHS)


# ============================================================================
# Step 7. Descriptive statistics (Tables 1 and 2)
# ============================================================================

def descriptives(v):
    o = v[OUTCOMES].dropna()
    t1 = o.agg(["mean", "std", "min", "max"]).T.round(0)
    t1["n"] = len(o)
    t1["SD of monthly change"] = o.diff().std().round(0)
    t1["month of min"] = o.idxmin().dt.strftime("%Y-%m")
    t1["month of max"] = o.idxmax().dt.strftime("%Y-%m")
    t1["share of months below zero"] = (o < 0).mean().round(3)
    t1.to_csv(TAB / "table1_outcomes.csv")

    cols = ["V", "D", "A"] + HALVES
    t2 = v[cols].describe().T[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]
    t2["SD of monthly change"] = v[cols].diff().std()
    t2["skewness"] = v[cols].skew()
    t2.round(3).to_csv(TAB / "table2_vda_descriptives.csv")


# ============================================================================
# Step 8. Face validity (Table 3)
# ============================================================================

def face_validity(v):
    rows = []
    for k in ["V", "D", "A", "AEB_mean"]:
        x = v[k]
        avg = x.rolling(5, min_periods=5).mean().shift(-3)
        hist = avg.dropna()
        for event, onset in EVENT_ONSETS:
            m = avg.get(pd.Timestamp(onset), np.nan)
            rows.append({"event": event, "onset": onset, "measure": k, "mean t-1 to t+3": m,
                         "percentile in sample": (hist < m).mean() * 100})
    f = pd.DataFrame(rows)
    t3 = f.pivot(index="event", columns="measure", values="mean t-1 to t+3").reindex([e for e, _ in EVENT_ONSETS])
    pct = f.pivot(index="event", columns="measure", values="percentile in sample").reindex(t3.index)
    t3 = t3[["V", "D", "A", "AEB_mean"]].round(2).join(pct[["V", "D", "A", "AEB_mean"]].round(0), rsuffix=" percentile")
    t3.loc["Baseline mean (2016-10 to 2019-12)"] = [v[k].loc[BASELINE[0]:BASELINE[1]].mean().round(2)
                                                    for k in ["V", "D", "A", "AEB_mean"]] + [np.nan] * 4
    t3.to_csv(TAB / "table3_face_validity.csv")


# ============================================================================
# Step 9. Convergent validity (Table 4)
# ============================================================================

def convergent_validity(v):
    rows = []
    for k in "VDA":
        j = v[[f"{k}_q", f"{k}_t"]].dropna()
        ann = j.groupby(j.index.year).mean()
        r, p = stats.pearsonr(j.iloc[:, 0], j.iloc[:, 1])
        rho, prho = stats.spearmanr(j.iloc[:, 0], j.iloc[:, 1])
        ra, pa = stats.pearsonr(ann.iloc[:, 0], ann.iloc[:, 1])
        rhoa, prhoa = stats.spearmanr(ann.iloc[:, 0], ann.iloc[:, 1])
        rows.append({"dimension": k, "n months": len(j), "pearson r": r, "p": p, "annual means r": ra, "p annual": pa,
                     "spearman rho": rho, "p rho": prho, "annual means rho": rhoa, "p rho annual": prhoa,
                     "n years": len(ann)})
    t4 = pd.DataFrame(rows).round(4)
    t4.to_csv(TAB / "table4_convergence.csv", index=False)
    return t4


# ============================================================================
# Step 10. Discriminant validity (Table 5)
# ============================================================================

def discriminant_validity(v):
    v[["V", "D", "A", "AEB_mean"]].corr().round(3).to_csv(TAB / "table5_discriminant.csv")
    v[HALVES + ["AEB_mean"]].corr().round(3).to_csv(TAB / "table5_submeasure_correlations.csv")


# ============================================================================
# Step 11. Predictive validity: joint Granger model (Table 6, Table A.4)
# ============================================================================

def stationarity(v):
    rows, trans = [], {}
    for name in OUTCOMES + ["V", "V_q", "V_t", "D", "D_q", "D_t", "A", "A_q", "A_t", "AEB_mean"]:
        s = v[name]
        adf = adfuller(s.dropna(), autolag="AIC")
        diff = adf[1] >= 0.05
        trans[name] = s.diff() if diff else s
        rows.append({"series": name, "n": len(s.dropna()), "ADF statistic": adf[0], "ADF p (levels)": adf[1],
                     "enters Granger tests as": "first difference" if diff else "levels",
                     "ADF p after differencing": adfuller(trans[name].dropna(), autolag="AIC")[1] if diff else np.nan})
    pd.DataFrame(rows).round(4).to_csv(TAB / "tableA4_stationarity.csv", index=False)
    return trans


def lags(s, p, name):
    return pd.concat({f"{name}_l{k}": s.shift(k) for k in range(1, p + 1)}, axis=1)


def lag_names(names, p):
    return [f"{n}_l{k}" for n in names for k in range(1, p + 1)]


def ols(d, y, x):
    return sm.OLS(d[y], sm.add_constant(d[x], has_constant="add")).fit()


def select_p(frame, names):
    full = pd.concat([frame.y] + [lags(frame[c], MAX_P, c) for c in names], axis=1).dropna()
    return min(range(1, MAX_P + 1), key=lambda p: ols(full, "y", lag_names(names, p)).aic)


def joint_model(tr, y, xs):
    names = ["y", "aeb"] + xs
    p = select_p(pd.DataFrame({"y": tr[y], "aeb": tr["AEB_mean"], **{x: tr[x] for x in xs}}), names)
    frame = pd.DataFrame({"y": tr[y], "aeb": tr["AEB_mean"], **{x: tr[x] / tr[x].std() for x in xs}})
    d = pd.concat([frame.y] + [lags(frame[c], p, c) for c in names], axis=1).dropna()
    fit = sm.OLS(d.y, sm.add_constant(d[lag_names(names, p)])).fit(cov_type="HAC", cov_kwds={"maxlags": p})
    return fit, p


def aeb_benchmark(tr):
    rows = []
    for y in OUTCOMES:
        frame = pd.DataFrame({"y": tr[y], "aeb": tr["AEB_mean"]})
        p = select_p(frame, ["y", "aeb"])
        d = pd.concat([frame.y, lags(frame.y, p, "y"), lags(frame.aeb, p, "aeb")], axis=1).dropna()
        w = sm.OLS(d.y, sm.add_constant(d[lag_names(["y", "aeb"], p)], has_constant="add")) \
            .fit(cov_type="HAC", cov_kwds={"maxlags": p}).f_test(", ".join(f"aeb_l{k} = 0" for k in range(1, p + 1)))
        rows.append({"outcome": y, "lags": p, "F": float(np.squeeze(w.fvalue)), "df_num": int(w.df_num),
                     "df_denom": int(w.df_denom), "p": float(w.pvalue)})
    pd.DataFrame(rows).round(4).to_csv(TAB / "aeb_alone_granger.csv", index=False)


def predictive_validity(v):
    tr = stationarity(v)
    tests, coefs = [], []
    for set_name, xs in SETS.items():
        for y in OUTCOMES:
            fit, p = joint_model(tr, y, xs)
            blocks = dict(zip(["V", "D", "A"], xs), AEB="aeb")
            for dim, x in blocks.items():
                w = fit.f_test(", ".join(f"{x}_l{k} = 0" for k in range(1, p + 1)))
                tests.append({"set": set_name, "outcome": y, "block": dim, "measure": x if x != "aeb" else "AEB_mean",
                              "lags": p, "F": float(np.squeeze(w.fvalue)), "df_num": int(w.df_num),
                              "df_denom": int(w.df_denom), "p": float(w.pvalue), "n": int(fit.nobs)})
                for k in range(1, p + 1):
                    c = f"{x}_l{k}"
                    coefs.append({"set": set_name, "outcome": y, "measure": x if x != "aeb" else "AEB_mean", "lag": k,
                                  "coef": fit.params[c], "HAC se": fit.bse[c], "p": fit.pvalues[c],
                                  "coef in SD of y": fit.params[c] / tr[y].std()})
    t = pd.DataFrame(tests)
    t["BH q"] = np.nan
    for _, idx in t.groupby("block").groups.items():
        t.loc[idx, "BH q"] = multipletests(t.loc[idx, "p"], method="fdr_bh")[1]
    t.round(4).to_csv(TAB / "table6_joint_granger_tests.csv", index=False)
    pd.DataFrame(coefs).round(4).to_csv(TAB / "table6_joint_granger_lags.csv", index=False)
    aeb_benchmark(tr)


# ============================================================================
# Step 12. Figures 2, 4 and 5
# ============================================================================

def plot_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 150, "figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": "#333333",
        "axes.grid": True, "grid.color": "#eeeeee", "grid.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "font.family": "serif", "font.size": 10, "axes.labelsize": 10,
        "axes.titlesize": 11, "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9, "lines.linewidth": 1.5})
    return plt


def event_lines(ax, alpha=0.35):
    for lbl, date in EVENT_LINES.items():
        d = pd.Timestamp(date)
        ax.axvline(d, color="#888888", linewidth=0.8, linestyle="--", alpha=alpha)
        top = ax.get_ylim()[1]
        ax.text(d, top * 0.97 if top > 0 else top, lbl, rotation=90, fontsize=7, color="#666666", ha="right", va="top")


def year_axis(ax):
    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def save(fig, name):
    import matplotlib.pyplot as plt
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def figures(v, conv):
    plt = plot_style()

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(v.index, v.AEB_mean, color="#2166ac", linewidth=1.5, label="Monthly AEB")
    ax.set_title("Purdue/CME Group Ag Economy Barometer")
    ax.set_ylabel("AEB Index")
    year_axis(ax); event_lines(ax)
    ax.legend(loc="lower left")
    plt.tight_layout()
    save(fig, "figure2_aeb")

    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True, gridspec_kw={"height_ratios": [1, 1, 1, 0.8]})
    for ax, k in zip(axes, "VDA"):
        s = v[k].dropna()
        ax.plot(s.index, s.values, color=COLORS[k], linewidth=1.5)
        ax.axhline(0, color="#999999", linewidth=0.7)
        ax.set_title(f"{k} -- {NAMES[k]}")
        ax.set_ylabel("Z-score")
        year_axis(ax); event_lines(ax, alpha=0.3)
    ax = axes[3]
    ref = v.AEB_mean.loc[BASELINE[0]:BASELINE[1]].mean()
    ax.plot(v.index, v.AEB_mean, color=COLORS["AEB"], linewidth=1.5)
    ax.axhline(ref, color="#999999", linewidth=0.7)
    ax.set_title(f"AEB mean -- Ag Economy Barometer (grey line = reference-period mean, {ref:.0f})")
    ax.set_ylabel("AEB index")
    year_axis(ax); event_lines(ax, alpha=0.3)
    fig.suptitle(f"VDA Dimensions -- Ag Economy Barometer ({v.index[0]:%Y/%m}-{v.index[-1]:%Y/%m})", fontsize=12, y=1.01)
    plt.tight_layout()
    save(fig, "figure4_vda_aeb")

    labels = {"V": ("Quant (rolling SD of ΔAEB)", "Text (rolling SD of concern churn)"),
              "D": ("Quant (cross-sect. SD of AEB)", "Text (pairwise topic distance, length-adj.)"),
              "A": ("Quant (rolling KL shape)", "Text (ambiguity-phrase share)")}
    fig, axes = plt.subplots(3, 1, figsize=(10, 9.5), sharex=True)
    for ax, k in zip(axes, "VDA"):
        ql, tl = labels[k]
        ax.plot(v.index, v[f"{k}_q"], color=COLORS[k], linewidth=1.5, label=ql, alpha=0.9)
        ax.plot(v.index, v[f"{k}_t"], color=COLORS[k], linewidth=1.5, label=tl, linestyle="--", alpha=0.7)
        ax.axhline(0, color="#999999", linewidth=0.7)
        r = conv.set_index("dimension").loc[k]
        ax.set_title(f"{k} -- {NAMES[k]}  (r = {r['pearson r']:.2f}, annual r = {r['annual means r']:.2f})")
        ax.set_ylabel("Z-score")
        year_axis(ax); event_lines(ax, alpha=0.25)
        ax.legend(loc="lower left", fontsize=8)
    fig.suptitle("VDA Sub-Measures: Quantitative vs Text-Based (Z-scored)", fontsize=12, y=1.01)
    plt.tight_layout()
    save(fig, "figure5_submeasures")


# ============================================================================
# Run
# ============================================================================

def main():
    for d in (TAB, FIG):
        d.mkdir(parents=True, exist_ok=True)
    v = build_monthly_series() if raw_data_available() else load_monthly_series()
    descriptives(v)
    face_validity(v)
    conv = convergent_validity(v)
    discriminant_validity(v)
    predictive_validity(v)
    figures(v, conv)
    print(f"Tables in {TAB}\nFigures in {FIG}")


if __name__ == "__main__":
    main()
