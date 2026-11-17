import re

import numpy as np
import pandas as pd
from scipy import stats

import mainanalysis as m

ROBUST_K = {4: 1, 5: 1, 6: 1, 8: 13}
TOPIC_LABELS = {
    "Trade and exports": {"trade", "export", "china"},
    "Economy, administration and immigration": {"economy", "president", "immigration"},
    "Regulation, interest rates and environment": {"regulation", "interest", "epa"},
    "Prices and input costs": {"cost", "input", "fuel"},
    "Crop and cattle markets": {"cattle", "corn", "weather"},
    "Livestock, meat and packers": {"beef", "meat", "packer"},
    "Taxes, land and family succession": {"tax", "family", "generation"},
}
ALT_BASELINES = {"2016-10 to 2018-09": ("2016-10-01", "2018-09-01"),
                 "2022-01 to 2023-12": ("2022-01-01", "2023-12-01"),
                 "full sample": (str(m.SAMPLE_START.date()), str(m.SAMPLE_END.date()))}


# ============================================================================
# Step 1. Rebuild the sub-measures from the raw data
# ============================================================================

def rebuild():
    ind = m.load_aeb_individual()
    sub, length = m.substantive_answers(m.load_answers())
    q = m.quantitative_measures(ind)
    sw = m.stop_words()
    dtm, vocab = m.document_term(sub.raw.tolist(), sw)
    jac, top = m.concern_churn(sub, sw)
    a_t_m = m.ambiguity_share(sub)
    t = pd.DataFrame({"V_t_m": jac, "V_t": jac.rolling(m.WINDOW, min_periods=m.WINDOW).std(),
                      "A_t_m": a_t_m, "A_t": m.roll(a_t_m)}, index=m.MONTHS)
    fits = {}
    for k, seed in {m.LDA_K: m.LDA_SEED, **ROBUST_K}.items():
        fits[k] = m.topic_distance(sub, length, dtm, k, seed)
    t["D_t"] = fits[m.LDA_K][0]
    return ind, sub, q, t, top, dtm, vocab, fits


# ============================================================================
# Step 2. Table A.1: concern churn example
# ============================================================================

def table_a1(t, top):
    vt = t.V_t.dropna()
    volatile = vt.idxmax()
    window = t.V_t_m.loc[volatile - pd.DateOffset(months=11):volatile]
    cols, notes = {}, []
    for month in [window.idxmin(), window.idxmax()]:
        prev = month - pd.DateOffset(months=1)
        a, b = top[prev], top[month]
        sa, sb = {w for w, _ in a}, {w for w, _ in b}
        cols[f"{prev:%Y-%m}"] = [f"{w} ({c})" + ("*" if w in sb else "") for w, c in a]
        cols[f"{month:%Y-%m}"] = [f"{w} ({c})" + ("*" if w in sa else "") for w, c in b]
        notes.append({"pair": f"{prev:%Y-%m} to {month:%Y-%m}", "shared": len(sa & sb), "union": len(sa | sb),
                      "Jaccard distance": round(1 - len(sa & sb) / len(sa | sb), 3)})
    pd.DataFrame({"rank": range(1, 21), **cols}).to_csv(m.TAB / "tableA1_jaccard_example.csv", index=False)
    pd.DataFrame(notes).to_csv(m.TAB / "tableA1_jaccard_distances.csv", index=False)


# ============================================================================
# Step 3. Table A.2: LDA topics and coherence
# ============================================================================

def coherence(dtm, idx, eps=1e-12):
    x = (dtm[:, idx] > 0).astype(float).toarray()
    p = x.mean(axis=0)
    pj = (x.T @ x) / x.shape[0]
    npmi = np.log((pj + eps) / np.outer(p, p)) / -np.log(pj + eps)
    np.fill_diagonal(npmi, 1.0)
    tot = npmi.sum(axis=0)
    cos = (npmi @ tot) / (np.linalg.norm(npmi, axis=1) * np.linalg.norm(tot))
    return float(npmi[np.triu_indices(len(idx), 1)].mean()), float(cos.mean())


def topic_table(lda, theta, usable, vocab, dtm, labels=None):
    idx = [np.argsort(c)[::-1][:10] for c in lda.components_]
    words = [list(vocab[i]) for i in idx]
    coh = [coherence(dtm, i) for i in idx]
    share = np.bincount(theta[usable].argmax(axis=1), minlength=len(idx)) / usable.sum()
    lab = []
    for w in words:
        hit = [k for k, anchors in (labels or {}).items() if len(anchors & set(w)) >= 2]
        lab.append(hit[0] if len(hit) == 1 else "")
    return pd.DataFrame({"topic": range(1, len(idx) + 1), "label": lab, "top ten words": [", ".join(w) for w in words],
                         "share of answers": share.round(3), "NPMI": [round(c[0], 3) for c in coh],
                         "C_v": [round(c[1], 3) for c in coh]})


def table_a2(fits, vocab, dtm):
    _, lda, theta, usable = fits[m.LDA_K]
    topic_table(lda, theta, usable, vocab, dtm, TOPIC_LABELS).to_csv(m.TAB / "tableA2_lda_topics.csv", index=False)


# ============================================================================
# Step 4. Table A.3: flagged answers with hedging phrases
# ============================================================================

def table_a3(sub):
    norm = m.ambiguity_norm(sub.raw)

    def phrase(text):
        for rx in m.AMBIG.values():
            hit = rx.search(text)
            if hit:
                return hit.group(0).replace("|", "").strip()
        return ""

    flagged = sub.assign(phrase=norm.apply(phrase))
    flagged = flagged[flagged.phrase != ""]
    sample = flagged.sample(n=15, random_state=m.SEED).sort_values("resp_id")
    sample.assign(month=sample.Date.dt.strftime("%Y-%m"))[["month", "raw", "phrase"]] \
        .rename(columns={"raw": "answer", "phrase": "matched phrase"}) \
        .to_csv(m.TAB / "tableA3_flagged_answers.csv", index=False)

    hedge = norm.apply(m.ambiguity_group) != ""
    rows = []
    for name, pat in m.UNRESTRICTED.items():
        used = norm.str.contains(re.compile(pat))
        rows.append({"phrase": name, "answers with the phrase": int(used.sum()),
                     "counted": int((used & hedge).sum()), "excluded": int((used & ~hedge).sum())})
    pd.DataFrame(rows).to_csv(m.TAB / "ambiguity_phrase_restrictions.csv", index=False)


# ============================================================================
# Step 5. Table A.5: number of LDA topics
# ============================================================================

def corr_with_annual(a, b):
    d = pd.concat([a, b], axis=1).dropna()
    ann = d.groupby(d.index.year).mean()
    return round(d.iloc[:, 0].corr(d.iloc[:, 1]), 3), round(ann.iloc[:, 0].corr(ann.iloc[:, 1]), 3)


def table_a5(q, fits, vocab, dtm):
    d_q = m.zscore(q.D_q)
    main = m.zscore(fits[m.LDA_K][0])
    rows = []
    for k in sorted(fits):
        d_t, lda, theta, usable = fits[k]
        tt = topic_table(lda, theta, usable, vocab, dtm)
        r, ra = corr_with_annual(d_q, m.zscore(d_t))
        rows.append({"K": k, "mean NPMI": round(tt.NPMI.mean(), 3), "mean C_v": round(tt.C_v.mean(), 3),
                     "r(D_q, D_t)": r, "r, annual means": ra, "corr with K = 7 D_t": round(main.corr(m.zscore(d_t)), 3)})
    pd.DataFrame(rows).to_csv(m.TAB / "tableA5_lda_k.csv", index=False)


# ============================================================================
# Step 6. Rolling window: 6 vs 12 months (Figure A.1)
# ============================================================================

def window_check(q, t):
    aeb_change = q.AEB_mean.diff()
    halves = {12: {"V_q": q.V_q, "V_t": t.V_t, "A_q": q.A_q, "A_t": t.A_t},
              6: {"V_q": aeb_change.rolling(6, min_periods=6).std(), "V_t": t.V_t_m.rolling(6, min_periods=6).std(),
                  "A_q": m.roll(q.A_kl_m, 6), "A_t": m.roll(t.A_t_m, 6)}}
    comp = {w: {k: (m.zscore(h[f"{k}_q"]) + m.zscore(h[f"{k}_t"])) / 2 for k in "VA"} for w, h in halves.items()}
    comp["D"] = (m.zscore(q.D_q) + m.zscore(t.D_t)) / 2
    pd.DataFrame([{"dimension": k, "corr of 6-month with 12-month composite": round(comp[6][k].corr(comp[12][k]), 3)}
                  for k in "VA"]).to_csv(m.TAB / "robustness_window.csv", index=False)

    plt = m.plot_style()
    fig, axes = plt.subplots(3, 1, figsize=(10, 9.5), sharex=True)
    for ax, k in zip(axes, "VDA"):
        if k == "D":
            ax.plot(m.MONTHS, comp["D"], color=m.COLORS[k], linewidth=1.5, label="monthly (both versions)")
        else:
            ax.plot(m.MONTHS, comp[12][k], color=m.COLORS[k], linewidth=1.6, label="12-month window (main)")
            ax.plot(m.MONTHS, comp[6][k], color=m.COLORS[k], linewidth=1.1, linestyle="--", alpha=0.8, label="6-month window")
        ax.axhline(0, color="#999999", linewidth=0.7)
        ax.set_title(f"{k} -- {m.NAMES[k]}")
        ax.set_ylabel("Z-score")
        m.year_axis(ax); m.event_lines(ax, alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("V, D, A composites: 12-month vs 6-month windows for V and A", fontsize=12, y=1.01)
    plt.tight_layout()
    m.save(fig, "figureA1_windows")


# ============================================================================
# Step 7. Reference period for the z-scores
# ============================================================================

def baseline_check(q, t):
    raw = {"V": (q.V_q, t.V_t), "D": (q.D_q, t.D_t), "A": (q.A_q, t.A_t)}
    main = {k: (m.zscore(a) + m.zscore(b)) / 2 for k, (a, b) in raw.items()}
    rows = []
    for name, base in ALT_BASELINES.items():
        alt = {k: (m.zscore(a, base) + m.zscore(b, base)) / 2 for k, (a, b) in raw.items()}
        rows.append({"reference period": name, **{f"corr {k} with main": round(alt[k].corr(main[k]), 3) for k in "VDA"}})
    pd.DataFrame(rows).to_csv(m.TAB / "robustness_baseline.csv", index=False)


# ============================================================================
# Step 8. Alternative shape statistics for A_q (Figure A.2)
# ============================================================================

def ambiguity_alternatives(ind, q, t):
    g = ind.groupby("Date").AEB
    sk = g.apply(lambda x: stats.skew(x.dropna(), bias=False))
    ku = g.apply(lambda x: stats.kurtosis(x.dropna(), bias=False))
    n = g.count()
    bc = (sk ** 2 + 1) / (ku + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))
    series = {"KL shape change (main)": m.zscore(q.A_q),
              "bimodality coefficient": m.zscore(m.roll(bc.reindex(m.MONTHS).diff().abs())),
              "excess kurtosis": m.zscore(m.roll(ku.reindex(m.MONTHS).diff().abs()))}
    a_t = m.zscore(t.A_t)
    rows = []
    for name, s in series.items():
        r, ra = corr_with_annual(s, a_t)
        rows.append({"A_q construction": name, "r(A_q, A_t)": r, "r, annual means": ra})
    pd.DataFrame(rows).to_csv(m.TAB / "robustness_ambiguity_alternatives.csv", index=False)

    plt = m.plot_style()
    styles = {"KL shape change (main)": ("#CC79A7", "-", 2.0), "bimodality coefficient": ("#0072B2", "--", 1.3),
              "excess kurtosis": ("#E69F00", ":", 1.5)}
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for name, (col, ls, lw) in styles.items():
        ax.plot(m.MONTHS, series[name], color=col, linestyle=ls, linewidth=lw, label=name)
    ax.axhline(0, color="#999999", linewidth=0.7)
    ax.set_title("Quantitative ambiguity (A_q): KL shape change vs alternative shape statistics, 12-month means")
    ax.set_ylabel("Z-score")
    ax.legend(fontsize=8, loc="upper right")
    m.year_axis(ax); m.event_lines(ax, alpha=0.3)
    plt.tight_layout()
    m.save(fig, "figureA2_kl_alternatives")


# ============================================================================
# Run
# ============================================================================

def main():
    if not m.raw_data_available():
        raise SystemExit("robustness.py needs the raw survey data in data/raw (see README).")
    for d in (m.TAB, m.FIG):
        d.mkdir(parents=True, exist_ok=True)
    ind, sub, q, t, top, dtm, vocab, fits = rebuild()
    table_a1(t, top)
    table_a2(fits, vocab, dtm)
    table_a3(sub)
    table_a5(q, fits, vocab, dtm)
    window_check(q, t)
    baseline_check(q, t)
    ambiguity_alternatives(ind, q, t)
    print(f"Tables in {m.TAB}\nFigures in {m.FIG}")


if __name__ == "__main__":
    main()
