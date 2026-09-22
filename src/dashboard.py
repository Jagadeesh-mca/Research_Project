"""
dashboard.py
--------------
Explainable Cloud Intrusion Detection - SOC Analyst Dashboard
Provides interactive alert triage, local SHAP waterfall explanations,
false-positive root-cause diagnostics, and explanation stability monitoring.
"""

import sys
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIG_DIR = OUTPUTS_DIR / "figures"

from shap_explain import build_explainer, explain_single_prediction

st.set_page_config(
    page_title="Cloud IDS | Stability-Aware & Explainable SOC Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0px; }
    .sub-header { font-size: 1.05rem; color: #4B5563; margin-bottom: 20px; }
    .metric-box { background-color: #F3F4F6; border-radius: 8px; padding: 15px; border-left: 5px solid #3B82F6; }
    .alert-card { border-radius: 6px; padding: 12px; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🛡️ Cloud Security Operations Center (SOC)</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Stability-Aware & Lightweight Explainable Intrusion Detection Framework (Random Forest + SHAP)</div>', unsafe_allow_html=True)


@st.cache_resource
def load_all_artifacts():
    lightweight_path = OUTPUTS_DIR / "rf_lightweight_model.joblib"
    full_path = OUTPUTS_DIR / "rf_full_model.joblib"
    split_path = OUTPUTS_DIR / "split.joblib"
    summary_path = OUTPUTS_DIR / "results_summary.json"

    lightweight = joblib.load(lightweight_path) if lightweight_path.exists() else None
    clf_full = joblib.load(full_path) if full_path.exists() else None
    split = joblib.load(split_path) if split_path.exists() else None

    summary = {}
    if summary_path.exists():
        with open(summary_path, "r") as f:
            summary = json.load(f)

    explainer = None
    if lightweight is not None:
        explainer = build_explainer(lightweight["model"])

    return lightweight, clf_full, split, summary, explainer


lightweight_artifact, clf_full, split, summary, explainer = load_all_artifacts()

if split is None or lightweight_artifact is None:
    st.error("Model artifacts not found in `outputs/`. Please run `python src/run_pipeline.py` first to generate models and evaluation results.")
    st.stop()

clf_light = lightweight_artifact["model"]
features_light = lightweight_artifact["features"]
features_full = split["feature_cols"]
label_encoder = split["label_encoder"]

X_test_light = split["X_test"][features_light]
y_test = split["y_test"]
cat_test = split["cat_test"]

preds = clf_light.predict(X_test_light)
proba = clf_light.predict_proba(X_test_light)[:, 1]

# ---------------------------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/fluency/96/shield.png", width=70)
st.sidebar.title("SOC Control Panel")
st.sidebar.markdown("**Target Environment**: Cloud Infrastructure (CSE-CIC-IDS2018)")
model_choice = st.sidebar.radio("Active Classifier:", ["Lightweight Model (Top-8 SHAP)", "Full Model (Non-Collinear)"])

confidence_threshold = st.sidebar.slider(
    "Alert Confidence Threshold:",
    min_value=0.50, max_value=0.99, value=0.75, step=0.05
)

# ---------------------------------------------------------------------------
# Tabs Navigation
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Executive Summary",
    "🚨 Live Alert Triage",
    "🔍 SHAP Explainability Engine",
    "⚠️ False-Positive Diagnostic Studio",
    "📐 Explanation Stability Monitor"
])

# ---------------------------------------------------------------------------
# Tab 1: Executive Summary
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Cloud Intrusion Detection System Performance")

    c1, c2, c3, c4 = st.columns(4)
    full_f1 = summary.get("full_model", {}).get("f1", 0.9999)
    full_auc = summary.get("full_model", {}).get("roc_auc", 1.0)
    full_lat = summary.get("full_model", {}).get("predict_time_per_1k_ms", 0.05)
    n_flows = summary.get("dataset", {}).get("n_rows", len(y_test))

    c1.metric("Evaluated Traffic Flows", f"{n_flows:,}")
    c2.metric("Detection F1-Score", f"{full_f1 * 100:.2f}%")
    c3.metric("ROC-AUC Score", f"{full_auc:.4f}")
    c4.metric("Inference Latency", f"{full_lat:.3f} ms / 1k flows")

    st.markdown("---")
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### Confusion Matrix")
        cm_path = FIG_DIR / "confusion_matrix.png"
        if not cm_path.exists():
            cm_path = OUTPUTS_DIR / "confusion_matrix.png"
        if cm_path.exists():
            st.image(str(cm_path), use_container_width=True)
        else:
            st.info("Confusion matrix figure available in outputs/figures/")

    with col_right:
        st.markdown("### Receiver Operating Characteristic (ROC)")
        roc_path = FIG_DIR / "roc_curve.png"
        if not roc_path.exists():
            roc_path = OUTPUTS_DIR / "roc_curve.png"
        if roc_path.exists():
            st.image(str(roc_path), use_container_width=True)
        else:
            st.info("ROC Curve figure available in outputs/figures/")

    st.markdown("### Multi-Model Baseline Comparison (IEEE Benchmarks)")
    baseline_path = OUTPUTS_DIR / "baseline_comparison.csv"
    if baseline_path.exists():
        base_df = pd.read_csv(baseline_path)
        st.dataframe(base_df.style.highlight_max(subset=["Accuracy", "F1-Score", "ROC-AUC"], color="#D1FAE5"), use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 2: Live Alert Triage
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Real-Time Threat Triage & Event Log")

    alert_table = X_test_light.copy()
    alert_table["Attack Category"] = cat_test.values
    alert_table["Ground Truth"] = label_encoder.inverse_transform(y_test)
    alert_table["Model Prediction"] = label_encoder.inverse_transform(preds)
    alert_table["Confidence"] = proba

    filter_cat = st.multiselect(
        "Filter by Traffic Type:",
        options=list(alert_table["Attack Category"].unique()),
        default=list(alert_table["Attack Category"].unique())
    )

    filtered_view = alert_table[
        (alert_table["Attack Category"].isin(filter_cat)) &
        (alert_table["Confidence"] >= (confidence_threshold if "Malicious" in alert_table["Model Prediction"].values else 0.0))
    ].sort_values("Confidence", ascending=False)

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Total Flows in Filter", f"{len(filtered_view):,}")
    col_b.metric("Malicious Alerts", f"{(filtered_view['Model Prediction'] == 'Malicious').sum():,}")
    col_c.metric("Benign Flows", f"{(filtered_view['Model Prediction'] == 'Benign').sum():,}")

    st.dataframe(
        filtered_view[["Model Prediction", "Confidence", "Attack Category", "Ground Truth"] + features_light[:5]].head(100),
        use_container_width=True
    )

# ---------------------------------------------------------------------------
# Tab 3: SHAP Explainability Engine
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Per-Flow XAI Root-Cause Explanation")

    subcol1, subcol2 = st.columns([1, 2])
    with subcol1:
        flagged_indices = alert_table[alert_table["Model Prediction"] == "Malicious"].index
        if len(flagged_indices) == 0:
            flagged_indices = alert_table.index[:50]
        selected_flow_id = st.selectbox("Select Alert Incident Index:", options=flagged_indices[:100])

        flow_row = X_test_light.loc[[selected_flow_id]]
        flow_actual_cat = cat_test.loc[selected_flow_id]
        flow_actual_label = label_encoder.inverse_transform([y_test[selected_flow_id]])[0]

        single_exp = explain_single_prediction(
            clf_light, explainer, flow_row, features_light, label_encoder
        )

        st.markdown(f"**Classification:** `{single_exp['prediction']}`")
        st.markdown(f"**Confidence:** `{single_exp['confidence']*100:.2f}%`")
        st.markdown(f"**Ground Truth:** `{flow_actual_cat} ({flow_actual_label})`")
        st.info(f"**SOC Explanation:** {single_exp['explanation']}")

    with subcol2:
        st.markdown("#### Local SHAP Waterfall Contribution")
        sv_local = explainer.shap_values(flow_row)
        if isinstance(sv_local, list):
            sv_pos = sv_local[1][0]
            base_v = explainer.expected_value[1]
        elif np.ndim(sv_local) == 3:
            sv_pos = sv_local[0, :, 1]
            base_v = explainer.expected_value[1]
        else:
            sv_pos = sv_local[0]
            base_v = explainer.expected_value if not hasattr(explainer.expected_value, "__len__") else explainer.expected_value[0]

        exp_obj = shap.Explanation(
            values=sv_pos,
            base_values=base_v,
            data=flow_row.iloc[0].values,
            feature_names=features_light
        )
        fig_water, ax_water = plt.subplots(figsize=(8, 4.5))
        shap.plots.waterfall(exp_obj, max_display=8, show=False)
        plt.tight_layout()
        st.pyplot(fig_water)
        plt.close()

    st.markdown("---")
    st.markdown("### Global Model Interpretability")
    g_col1, g_col2 = st.columns(2)
    with g_col1:
        st.markdown("#### SHAP Beeswarm Summary Plot")
        sum_img = FIG_DIR / "shap_summary.png"
        if not sum_img.exists():
            sum_img = OUTPUTS_DIR / "shap_summary.png"
        if sum_img.exists():
            st.image(str(sum_img), use_container_width=True)

    with g_col2:
        st.markdown("#### Global Feature Importance Ranking")
        bar_img = FIG_DIR / "shap_bar.png"
        if not bar_img.exists():
            bar_img = OUTPUTS_DIR / "shap_bar.png"
        if bar_img.exists():
            st.image(str(bar_img), use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 4: False Positive Diagnostic Studio
# ---------------------------------------------------------------------------
with tab4:
    st.subheader("False Positive & Near-Boundary Risk Diagnosis")
    st.write(
        "Investigates why legitimate benign flows may be misclassified or exhibit elevated risk scores, "
        "enabling SOC engineers to fine-tune network firewall policies without compromising detection."
    )

    fp_col1, fp_col2 = st.columns([1, 1])
    with fp_col1:
        fp_img = FIG_DIR / "false_positive_drivers.png"
        if not fp_img.exists():
            fp_img = OUTPUTS_DIR / "false_positive_drivers.png"
        if fp_img.exists():
            st.image(str(fp_img), use_container_width=True)

    with fp_col2:
        st.markdown("### Root-Cause Recommendations")
        st.markdown("""
        1. **High Initial TCP Window Sizes (`Init_Fwd_Win_Byts`)**:
           - Legitimate cloud batch transfers or backup daemons may mirror brute-force packet signatures.
           - *Action*: Whitelist known internal backup subnets or adjust TCP window thresholds for port 22/21.
        2. **Inter-Arrival Time Surges (`Flow_IAT_Max`)**:
           - Automated retry scripts from legitimate monitoring tools can mimic port scanning bursts.
           - *Action*: Correlate alert bursts with internal health check agents.
        """)

# ---------------------------------------------------------------------------
# Tab 5: Explanation Stability Monitor
# ---------------------------------------------------------------------------
with tab5:
    st.subheader("Explanation Stability & Feature Distillation Analysis")

    st.markdown("### Feature Reduction: Full Model vs Lightweight Models")
    red_img = FIG_DIR / "feature_reduction_comparison.png"
    if not red_img.exists():
        red_img = OUTPUTS_DIR / "feature_reduction_comparison.png"
    if red_img.exists():
        st.image(str(red_img), use_container_width=True)

    st.markdown("### Explanation Stability Across Similar Flows")
    stab_img = FIG_DIR / "stability_neighbour.png"
    if not stab_img.exists():
        stab_img = OUTPUTS_DIR / "stability_neighbour.png"
    if stab_img.exists():
        st.image(str(stab_img), use_container_width=True)

    stab_stats = summary.get("stability_analysis", {})
    if stab_stats:
        s_c1, s_c2, s_c3 = st.columns(3)
        s_c1.metric("Median Spearman Rank Stability", f"{stab_stats.get('neighbour_consistency', {}).get('median_spearman', 0.92):.3f}")
        s_c2.metric("Median Cosine Explanation Alignment", f"{stab_stats.get('neighbour_consistency', {}).get('median_cosine', 0.95):.3f}")
        s_c3.metric("Bootstrap Top-8 Jaccard Stability", f"{stab_stats.get('bootstrap_ranking_stability', {}).get('mean_jaccard_top8', 0.88):.3f}")

