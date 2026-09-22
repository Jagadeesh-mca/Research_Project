"""
shap_explain.py
-----------------
SHAP explanation stage -> Identify important traffic features.

Uses TreeExplainer (exact, fast for Random Forest) rather than
KernelExplainer, since the model is tree-based.
"""

from pathlib import Path

import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def build_explainer(clf):
    return shap.TreeExplainer(clf)


def compute_shap_values(explainer, X, sample_size=2000, seed=42):
    """
    SHAP on the full test set is expensive; sample for interactive/plot
    purposes but keep the sample stratified-ish by just random sampling
    (the test set is already stratified by class).
    """
    if len(X) > sample_size:
        X_sample = X.sample(sample_size, random_state=seed)
    else:
        X_sample = X
    sv = explainer.shap_values(X_sample)
    # sklearn RF binary classifier -> shap returns array shaped
    # (n_samples, n_features, n_classes) in recent shap versions, or a list
    # of two arrays in older ones. Normalise to "malicious class" values.
    if isinstance(sv, list):
        sv_pos = sv[1]
    elif sv.ndim == 3:
        sv_pos = sv[:, :, 1]
    else:
        sv_pos = sv
    return sv_pos, X_sample


def global_feature_importance(shap_values, feature_names):
    mean_abs = np.abs(shap_values).mean(axis=0)
    ranking = pd.Series(mean_abs, index=feature_names).sort_values(ascending=False)
    return ranking


def plot_summary(shap_values, X_sample, out_path, show_plot=True):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_sample, show=False, max_display=15)
    plt.title("SHAP Beeswarm Summary Plot (Top 15 Features)", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_bar(ranking, out_path, top_n=15, show_plot=True):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 6.5))
    top_series = ranking.head(top_n).sort_values()
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_series)))
    bars = plt.barh(top_series.index, top_series.values, color=colors, edgecolor="black", alpha=0.85)
    plt.xlabel("Mean |SHAP Value| (Contribution toward Attack Classification)", fontsize=11, fontweight="bold")
    plt.title(f"Top {top_n} Most Influential Network Flow Features", fontsize=12, fontweight="bold", pad=12)
    plt.grid(axis="x", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_dependence(shap_values, X_sample, feature_name, out_path, show_plot=True):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    if feature_name not in X_sample.columns:
        return
    plt.figure(figsize=(8, 5))
    shap.dependence_plot(feature_name, shap_values, X_sample, show=False)
    plt.title(f"SHAP Dependence Plot: {feature_name}", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_waterfall(explainer, X_row, feature_names, out_path, title="Local SHAP Explanation: Feature Contributions for Flow", show_plot=True):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    sv = explainer.shap_values(X_row)
    if isinstance(sv, list):
        sv_pos = sv[1][0]
        base_val = explainer.expected_value[1]
    elif np.ndim(sv) == 3:
        sv_pos = sv[0, :, 1]
        base_val = explainer.expected_value[1]
    else:
        sv_pos = sv[0]
        base_val = explainer.expected_value if not hasattr(explainer.expected_value, "__len__") else explainer.expected_value[0]

    # Create Explanation object for shap waterfall
    exp = shap.Explanation(
        values=sv_pos,
        base_values=base_val,
        data=X_row.iloc[0].values,
        feature_names=feature_names
    )
    plt.figure(figsize=(9, 6))
    shap.plots.waterfall(exp, max_display=10, show=False)
    plt.title(title, fontsize=11, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    if show_plot:
        plt.show()
    else:
        plt.close()


def explain_single_prediction(clf, explainer, X_row, feature_names, label_encoder):
    """
    Produces the analyst-facing explanation shown in the proposal's
    'Example output' block: prediction, confidence, top features, and a
    plain-language sentence.
    """
    proba = clf.predict_proba(X_row)[0]
    pred_idx = int(np.argmax(proba))
    pred_label = label_encoder.classes_[pred_idx]
    confidence = proba[pred_idx]

    sv = explainer.shap_values(X_row)
    if isinstance(sv, list):
        sv_pos = sv[1][0]
    elif np.ndim(sv) == 3:
        sv_pos = sv[0, :, 1]
    else:
        sv_pos = sv[0]

    contrib = pd.Series(sv_pos, index=feature_names).sort_values(key=np.abs, ascending=False)
    top_features = contrib.head(4)

    direction = "increase" if pred_label == "Malicious" else "decrease"
    explanation_sentence = (
        f"The traffic is classified as {pred_label.lower()} primarily because of its "
        f"{', '.join(top_features.index[:3]).lower().replace('_', ' ')}, which {direction} "
        f"the model's confidence relative to typical traffic."
    )

    return {
        "prediction": pred_label,
        "confidence": confidence,
        "top_features": top_features,
        "explanation": explanation_sentence,
    }


if __name__ == "__main__":
    clf = joblib.load(OUTPUTS_DIR / "rf_full_model.joblib")
    split = joblib.load(OUTPUTS_DIR / "split.joblib")

    explainer = build_explainer(clf)
    shap_values, X_sample = compute_shap_values(explainer, split["X_test"], sample_size=1500)

    ranking = global_feature_importance(shap_values, split["feature_cols"])
    print("--- Top 10 features by mean |SHAP value| ---")
    print(ranking.head(10))

    plot_summary(shap_values, X_sample, OUTPUTS_DIR / "shap_summary.png")
    plot_bar(ranking, OUTPUTS_DIR / "shap_bar.png")
    ranking.to_csv(OUTPUTS_DIR / "shap_feature_ranking.csv")

    # Example single-prediction explanation, mirroring the proposal's format.
    # Pick a confidently, correctly predicted malicious flow so the example
    # is representative rather than an edge case.
    malicious_idx = split["X_test"][split["y_test"] == 1].index
    proba_malicious = clf.predict_proba(split["X_test"].loc[malicious_idx])[:, 1]
    best_idx = malicious_idx[np.argmax(proba_malicious)]
    example_row = split["X_test"].loc[[best_idx]]
    example_category = split["cat_test"].loc[best_idx]
    result = explain_single_prediction(
        clf, explainer, example_row, split["feature_cols"], split["label_encoder"]
    )
    print("\n--- Example analyst-facing explanation ---")
    print(f"Prediction: {result['prediction']} traffic")
    print(f"Attack category (ground truth): {example_category}")
    print(f"Confidence: {result['confidence']*100:.0f}%")
    print("Important features:")
    for f, v in result["top_features"].items():
        print(f"  - {f} (SHAP={v:+.3f})")
    print(f"Analyst explanation: {result['explanation']}")

    joblib.dump({"explainer": explainer, "ranking": ranking},
                OUTPUTS_DIR / "shap_artifacts.joblib")
