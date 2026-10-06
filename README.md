# A Stability-Aware and Lightweight Explainable Intrusion Detection Framework
### Random Forest + SHAP for Cloud Security

This is a complete, working implementation of the pipeline described in the
project proposal. Every stage runs end-to-end via `src/run_pipeline.py`.

## Important: about the dataset

The real **CSE-CIC-IDS2018** dataset (~6-16 GB, hosted on AWS S3 by the
Canadian Institute for Cybersecurity, UNB) could not be downloaded inside
this build environment's sandboxed network. To let every module be built,
run, and validated end-to-end, `src/generate_demo_data.py` produces a
**synthetic dataset that replicates CIC-IDS2018's exact column schema,
attack-category mix, and ~90/10 benign/malicious class imbalance.**

**All numbers in `outputs/results_summary.json` and the accompanying paper
are proof-of-concept results from this synthetic data** — they demonstrate
that the pipeline (cleaning -> RF -> SHAP -> false-positive analysis ->
feature reduction -> stability testing -> dashboard) works correctly, but
they are **not a claim about real-world detection performance**.

### To run this on the real dataset
1. Download the CSE-CIC-IDS2018 CSVs from
   https://www.unb.ca/cic/datasets/ids-2018.html (or via the AWS CLI command
   on that page) to a machine with outbound internet access.
2. In `src/generate_demo_data.py`, use `load_real_dataset(csv_paths)`
   instead of `generate()` — it normalises the real column names
   (e.g. `" Flow Duration"` -> `Flow_Duration`) to match every other module.
3. Re-run `python src/run_pipeline.py`. No other code changes are needed;
   every downstream module (preprocessing, training, SHAP, false-positive
   analysis, feature reduction, stability testing, dashboard) is written
   against the column schema, not the data source.

## Project structure

```
ids_project/
├── README.md
├── data/
│   └── demo_traffic.csv              # synthetic proof-of-concept dataset
├── src/
│   ├── generate_demo_data.py         # synthetic data / real-data loader
│   ├── preprocessing.py              # cleaning, imputation, train/test split
│   ├── train_model.py                # Random Forest training + evaluation
│   ├── shap_explain.py               # global + per-prediction SHAP explanations
│   ├── false_positive_analysis.py    # why benign traffic gets misflagged
│   ├── feature_reduction.py          # SHAP-driven lightweight model
│   ├── stability_analysis.py         # neighbour + bootstrap stability tests
│   ├── dashboard.py                  # Streamlit analyst dashboard (optional)
│   └── run_pipeline.py               # runs everything end-to-end
└── outputs/
    ├── results_summary.json          # single source of truth for all results
    ├── rf_full_model.joblib / rf_lightweight_model.joblib
    ├── shap_summary.png / shap_bar.png / shap_feature_ranking.csv
    ├── false_positive_drivers.png
    ├── feature_reduction_comparison.png / .csv
    └── stability_neighbour.png
```

## How to run

**On the synthetic proof-of-concept data (default):**
```bash
cd src
python3 run_pipeline.py          # runs all 9 stages, ~3-6 minutes
streamlit run dashboard.py       # optional: interactive alert dashboard
```

To run the pipeline without opening plot windows (for example, if Tkinter
reports GUI cleanup errors), use:
```bash
python run_pipeline.py --mode 2 --no-gui
```
Plots are still saved under `outputs/figures/`; Matplotlib uses a non-GUI
backend for this run.

**On the real CSE-CIC-IDS2018 dataset:**
Open `src/run_pipeline.py`, find the `DATA SOURCE CONFIGURATION` block near
the top, and set:
```python
REAL_DATA_PATHS = [r"C:\Users\you\Downloads\02-14-2018.csv"]
```
(list more than one day's CSV in the same list if you have them). Then run
`python3 run_pipeline.py` exactly as above — no other code changes needed.
`load_real_dataset()` in `generate_demo_data.py` handles the real files'
quirks automatically: multi-class labels (FTP-BruteForce, SSH-Bruteforce,
etc. all folded into a binary Benign/Malicious target, with the original
attack name kept as `Attack_Category`), duplicated header rows that appear
mid-file in the official CSVs, and dropping Source/Destination IP,
Flow ID, and Timestamp to prevent the model from memorising specific hosts
instead of learning traffic behaviour.

Real daily files can be 500k-1M+ rows; Random Forest + SHAP over the full
file works but is slow. If a first run is taking too long, set
`SAMPLE_SIZE = 200_000` (or similar) in the same config block to subsample.

## What each stage does (maps directly to the proposal)

| Proposal step | Module | Output |
|---|---|---|
| Data cleaning and preprocessing | `preprocessing.py` | deduplicated, imputed, stratified train/test split |
| Correlation-based redundancy removal | `correlation_analysis.py` | drops features correlated above 0.90, correlation heatmap |
| Feature selection / Random Forest classifier | `train_model.py` | trained RF, accuracy/precision/recall/F1/ROC-AUC |
| SHAP explanation | `shap_explain.py` | global feature ranking, summary plot, per-alert explanation |
| False-positive analysis | `false_positive_analysis.py` | which features drive benign flows into false alerts |
| SHAP-driven feature reduction | `feature_reduction.py` | full vs. top-5/8/12-feature model comparison |
| Explanation stability | `stability_analysis.py` | neighbour-consistency + bootstrap-retrain ranking stability |
| Lightweight model and alert dashboard | `dashboard.py` | Streamlit UI over the reduced model |

Note: on the synthetic dataset, correlation-based removal drops 0 features
(the generator's features happen to be fairly independent by construction).
On real CIC-IDS2018 data — which has well-documented near-duplicate
features (e.g. multiple packet-length and byte-count variants derived
from the same underlying counters) — expect this stage to actually remove
several redundant columns before SHAP-driven reduction runs on what's left.

## Headline proof-of-concept results (synthetic data)

- Full model (23 features): F1 = 0.919, ROC-AUC = 0.992
- Lightweight model (top-8 SHAP features): F1 = 0.924, ROC-AUC = 0.990,
  ~7% faster inference, 35% of the original feature set
- Explanation stability: median neighbour-explanation correlation = 0.89;
  84% of similar-traffic pairs exceed 0.7 correlation
- Bootstrap ranking stability across retrains: mean Jaccard overlap of
  top-8 features = 0.85, mean Spearman correlation of full ranking = 0.96
- False positives: 0.25% of test flows; ~39% of those show elevated
  SYN-flag/packet-count patterns resembling scanning traffic rather than
  random error

See `outputs/results_summary.json` for full detail.
"# Research_Project" 
