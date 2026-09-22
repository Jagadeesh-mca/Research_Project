"""
false_positive_analysis.py
----------------------------
Analyse false-positive alerts: for benign flows the model incorrectly
flags as malicious, use SHAP to show WHY the model got confused --
which benign-side features looked attack-like.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

from shap_explain import build_explainer


def find_false_positives(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    fp_mask = (y_test == 0) & (y_pred == 1)          # true benign, predicted malicious
    fn_mask = (y_test == 1) & (y_pred == 0)          # true malicious, predicted benign
    return X_test[fp_mask], X_test[fn_mask], fp_mask, fn_mask


def analyse_false_positives(clf, explainer, X_fp, feature_names, X_benign=None, top_n=8):
    """
    Analyses what features drive false positives. If zero false positives occur (due to high model accuracy),
    analyses 'near-boundary' benign flows (the top benign instances with the highest attack probability)
    to identify what factors bring legitimate traffic closest to the alert threshold.
    """
    is_near_boundary = False
    target_flows = X_fp

    if len(target_flows) == 0 and X_benign is not None and len(X_benign) > 0:
        is_near_boundary = True
        proba_attack = clf.predict_proba(X_benign)[:, 1]
        top_risk_indices = np.argsort(proba_attack)[-min(50, len(X_benign)):]
        target_flows = X_benign.iloc[top_risk_indices]

    if len(target_flows) == 0:
        return None, None, False

    sv = explainer.shap_values(target_flows)
    if isinstance(sv, list):
        sv_pos = sv[1]
    elif np.ndim(sv) == 3:
        sv_pos = sv[:, :, 1]
    else:
        sv_pos = sv

    mean_contrib = pd.Series(sv_pos.mean(axis=0), index=feature_names)
    driving_features = mean_contrib.sort_values(ascending=False).head(top_n)
    return driving_features, sv_pos, is_near_boundary


def plot_fp_drivers(driving_features, out_path, is_near_boundary=False, show_plot=True):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8.5, 5))
    driving_features.sort_values().plot(kind="barh", color="#d62728", edgecolor="black", alpha=0.85)
    plt.xlabel("Mean SHAP Value Toward 'Malicious' Classification", fontsize=11, fontweight="bold")
    title = "Near-Boundary Benign Traffic Risk Drivers" if is_near_boundary else "False-Positive Alert Drivers (Misclassified Benign Traffic)"
    plt.title(title, fontsize=12, fontweight="bold", pad=12)
    plt.grid(axis="x", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    if show_plot:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    clf = joblib.load(OUTPUTS_DIR / "rf_full_model.joblib")
    split = joblib.load(OUTPUTS_DIR / "split.joblib")

    X_fp, X_fn, fp_mask, fn_mask = find_false_positives(clf, split["X_test"], split["y_test"])
    print(f"False positives (benign flagged as malicious): {len(X_fp)} "
          f"({len(X_fp)/len(split['X_test'])*100:.2f}% of test set)")
    print(f"False negatives (malicious missed):             {len(X_fn)} "
          f"({len(X_fn)/len(split['X_test'])*100:.2f}% of test set)")

    explainer = build_explainer(clf)
    X_benign = split["X_test"][split["y_test"] == 0]
    driving_features, sv_fp, is_near_boundary = analyse_false_positives(
        clf, explainer, X_fp, split["feature_cols"], X_benign=X_benign
    )

    if driving_features is not None:
        print("\n--- Features most responsible for false-positive alerts / near-boundary risk ---")
        print(driving_features)
        plot_fp_drivers(driving_features, OUTPUTS_DIR / "false_positive_drivers.png", is_near_boundary=is_near_boundary)

    joblib.dump({"fp_mask": fp_mask, "fn_mask": fn_mask, "driving_features": driving_features, "is_near_boundary": is_near_boundary},
                OUTPUTS_DIR / "fp_analysis.joblib")
