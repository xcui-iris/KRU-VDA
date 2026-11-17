# Decomposing Knightian Risk and Uncertainty in Producers' Subjective Beliefs

Replication code for the paper. Producers' uncertainty is measured in three dimensions, each built from the
Purdue University/CME Group Ag Economy Barometer (AEB), October 2015 to September 2025:

- **Volatility (V)**
- **Dispersion (D)**
- **Ambiguity (A)**

Each dimension averages a quantitative sub-measure, built from the survey scores, and a text sub-measure, built from
the open-ended answers.

## Files

| File | Contents |
|---|---|
| `mainanalysis.py` | Builds the measures, runs the four validation tests (face, convergent, discriminant, predictive), and produces Tables 1–6, Table A.4 and Figures 2, 4, 5 |
| `robustness.py` | Appendix and robustness checks: Tables A.1, A.2, A.3, A.5, Figures A.1, A.2, and the robustness results in the text |
| `data/vda_monthly.csv` | Monthly series used in all four validation tests (see below) |
| `requirements.txt` | Python packages |

## Data

`data/vda_monthly.csv` has one row per month, October 2015 to September 2025.

| Column | Description |
|---|---|
| `date` | Month |
| `AEB_mean` | Ag Economy Barometer index (monthly mean of individual scores) |
| `V`, `D`, `A` | Composite dimensions (z-scores relative to October 2016 to December 2019) |
| `V_q`, `D_q`, `A_q` | Quantitative sub-measures (z-scores) |
| `V_t`, `D_t`, `A_t` | Text sub-measures (z-scores) |
| `equipment_shipments` | Farm machinery shipments, FRED A33AVS ($ million, seasonally adjusted) |
| `equipment_inventories` | Farm machinery inventories, FRED U33ATI ($ million, seasonally adjusted with STL) |
| `cot_net_commercial` | CFTC commercial long minus short, corn and soybeans, last report of the month (contracts) |
| `cot_hedge_turnover` | Mean weekly \|change in commercial long\| + \|change in commercial short\|, corn and soybeans (contracts per week, seasonally adjusted with STL) |

V and A start in October 2016 because they use 12-month windows.

The individual survey responses and open-ended answers are not shared. The two scripts handle them as follows:

- **`mainanalysis.py`** reads `data/vda_monthly.csv` when the raw data are absent. It reproduces every validation table and Figures 2, 4 and 5.
- **`robustness.py`** rebuilds the sub-measures from the raw data, so it needs them.

To run with the raw data, place the files under `data/raw/`, or point the `KRU_RAW_DATA` environment variable to their folder:

```
data/raw/
  historical aeb data - malone.xlsx        individual AEB responses
  Word Cloud Data/word cloud - YY-MM.xlsx  open-ended answers, one workbook per month
  equipment_price_index/monthly.csv        FRED A33AVS
  equipment_price_index/monthly,_end_of_period.csv   FRED U33ATI
  cot/dea_fut_xls_YYYY/annual.xls          CFTC Legacy COT, futures only
```

With the raw data present, `mainanalysis.py` rebuilds the measures and rewrites `data/vda_monthly.csv`.

## Running

```
pip install -r requirements.txt
python -m nltk.downloader stopwords
python mainanalysis.py
python robustness.py
```

Tables are written to `output/tables/` and figures to `output/figures/`. `mainanalysis.py` takes about 2 minutes with the raw data and seconds without them. `robustness.py` takes about 3 minutes.

## Output

| Paper | File |
|---|---|
| Table 1 | `table1_outcomes.csv` |
| Table 2 | `table2_vda_descriptives.csv` |
| Table 3 | `table3_face_validity.csv` |
| Table 4 | `table4_convergence.csv` |
| Table 5 | `table5_discriminant.csv`, `table5_submeasure_correlations.csv` |
| Table 6 | `table6_joint_granger_tests.csv`, `table6_joint_granger_lags.csv`, `aeb_alone_granger.csv` |
| Table A.1 | `tableA1_jaccard_example.csv`, `tableA1_jaccard_distances.csv` |
| Table A.2 | `tableA2_lda_topics.csv` |
| Table A.3 | `tableA3_flagged_answers.csv`, `ambiguity_phrase_restrictions.csv` |
| Table A.4 | `tableA4_stationarity.csv` |
| Table A.5 | `tableA5_lda_k.csv` |
| Robustness (text) | `robustness_window.csv`, `robustness_baseline.csv`, `robustness_ambiguity_alternatives.csv` |
| Figures 2, 4, 5 | `figure2_aeb`, `figure4_vda_aeb`, `figure5_submeasures` |
| Figures A.1, A.2 | `figureA1_windows`, `figureA2_kl_alternatives` |

Figures 1 and 3 are conceptual diagrams and are not produced by code.
