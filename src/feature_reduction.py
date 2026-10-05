"""
feature_reduction.py
----------------------
Reduce unnecessary features to create a lightweight model, then compare
full-feature vs reduced-feature performance and efficiency.
"""

import time
import sys
from pathlib import Path

import joblib
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

from train_model import train_random_forest, evaluate


def build_reduced_dataset(split, ranking, k):
    top_features = ranking.head(k).index.tolist()
    return {
        **split,
        "X_train": split["X_train"][top_features],
        "X_test": split["X_test"][top_features],
        "feature_cols": top_features,
    }, top_features


def compare_full_vs_reduced(split, ranking, k_values=(5, 8, 10, 12, 15)):
    rows = []

    # Full model baseline
    clf_full, train_time_full = train_random_forest(split["X_train"], split["y_train"])
    res_full = evaluate(clf_full, split["X_test"], split["y_test"], split["label_encoder"])
    rows.append({
        "model": f"Full ({len(split['feature_cols'])} features)",
        "n_features": len(split["feature_cols"]),
        "accuracy": res_full["metrics"]["accuracy"],
        "precision": res_full["metrics"]["precision"],
        "recall": res_full["metrics"]["recall"],
        "f1": res_full["metrics"]["f1"],
        "roc_auc": res_full["metrics"]["roc_auc"],
        "fpr": res_full["metrics"]["false_positive_rate"],
        "fnr": res_full["metrics"]["false_negative_rate"],
        "train_time_s": train_time_full,
        "predict_time_s": res_full["metrics"]["predict_time_s"],
        "latency_per_flow_ms": res_full["metrics"]["latency_per_flow_ms"],
        "latency_per_1k_ms": res_full["metrics"]["latency_per_1k_flows_ms"],
        "throughput_flows_s": res_full["metrics"]["throughput_flows_per_s"],
    })

    models = {"full": (clf_full, res_full, split["feature_cols"])}

    for k in k_values:
        if k > len(ranking):
            continue
        reduced_split, top_features = build_reduced_dataset(split, ranking, k)
        clf_k, train_time_k = train_random_forest(reduced_split["X_train"], reduced_split["y_train"])
        res_k = evaluate(clf_k, reduced_split["X_test"], split["y_test"], split["label_encoder"])
        rows.append({
            "model": f"Top-{k} features",
            "n_features": k,
            "accuracy": res_k["metrics"]["accuracy"],
            "precision": res_k["metrics"]["precision"],
            "recall": res_k["metrics"]["recall"],
            "f1": res_k["metrics"]["f1"],
            "roc_auc": res_k["metrics"]["roc_auc"],
            "fpr": res_k["metrics"]["false_positive_rate"],
            "fnr": res_k["metrics"]["false_negative_rate"],
            "train_time_s": train_time_k,
            "predict_time_s": res_k["metrics"]["predict_time_s"],
            "latency_per_flow_ms": res_k["metrics"]["latency_per_flow_ms"],
            "latency_per_1k_ms": res_k["metrics"]["latency_per_1k_flows_ms"],
            "throughput_flows_s": res_k["metrics"]["throughput_flows_per_s"],
        })
        models[f"top_{k}"] = (clf_k, res_k, top_features)

    return pd.DataFrame(rows), models


def plot_comparison(df, out_path, show_plot=True):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # 1. F1-Score
    axes[0, 0].bar(df["model"], df["f1"], color="#1f77b4", edgecolor="black", alpha=0.85)
    axes[0, 0].set_ylabel("F1-Score", fontsize=11, fontweight="bold")
    axes[0, 0].set_title("Detection Performance (F1-Score)", fontsize=11, fontweight="bold")
    axes[0, 0].set_ylim(min(df["f1"].min() * 0.95, 0.85), 1.02)
    axes[0, 0].tick_params(axis="x", rotation=20)
    axes[0, 0].grid(axis="y", linestyle=":", alpha=0.6)

    # 2. Inference Latency
    axes[0, 1].bar(df["model"], df["latency_per_1k_ms"], color="#ff7f0e", edgecolor="black", alpha=0.85)
    axes[0, 1].set_ylabel("Inference Latency (ms / 1k flows)", fontsize=11, fontweight="bold")
    axes[0, 1].set_title("Inference Latency (Lower is Better)", fontsize=11, fontweight="bold")
    axes[0, 1].tick_params(axis="x", rotation=20)
    axes[0, 1].grid(axis="y", linestyle=":", alpha=0.6)

    # 3. Throughput
    axes[1, 0].bar(df["model"], df["throughput_flows_s"] / 1000.0, color="#2ca02c", edgecolor="black", alpha=0.85)
    axes[1, 0].set_ylabel("Throughput (kFlows / sec)", fontsize=11, fontweight="bold")
    axes[1, 0].set_title("Detection Throughput (Higher is Better)", fontsize=11, fontweight="bold")
    axes[1, 0].tick_params(axis="x", rotation=20)
    axes[1, 0].grid(axis="y", linestyle=":", alpha=0.6)

    # 4. Feature Count vs Throughput Trade-off
    axes[1, 1].plot(df["n_features"], df["throughput_flows_s"] / 1000.0, marker="o", color="#d62728", lw=2, markersize=8)
    for _, row in df.iterrows():
        axes[1, 1].annotate(f"{row['n_features']} feat\n({row['f1']:.4f} F1)", 
                            (row["n_features"], row["throughput_flows_s"] / 1000.0),
                            textcoords="offset points", xytext=(0, 10), ha="center", fontsize=8, fontweight="bold")
    axes[1, 1].set_xlabel("Number of Features", fontsize=11, fontweight="bold")
    axes[1, 1].set_ylabel("Throughput (kFlows / sec)", fontsize=11, fontweight="bold")
    axes[1, 1].set_title("Feature Economy vs Throughput Trade-off", fontsize=11, fontweight="bold")
    axes[1, 1].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    if show_plot:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    split = joblib.load(OUTPUTS_DIR / "split.joblib")
    shap_artifacts = joblib.load(OUTPUTS_DIR / "shap_artifacts.joblib")
    ranking = shap_artifacts["ranking"]

    comparison_df, models = compare_full_vs_reduced(split, ranking, k_values=(5, 8, 12))
    print(comparison_df.to_string(index=False))

    plot_comparison(comparison_df, OUTPUTS_DIR / "feature_reduction_comparison.png")
    comparison_df.to_csv(OUTPUTS_DIR / "feature_reduction_comparison.csv", index=False)

    # Save the best lightweight model (top-8 chosen as the reported
    # "lightweight model" -- good F1 retention with roughly a third of
    # the original features)
    best_clf, best_res, best_features = models["top_8"]
    joblib.dump({"model": best_clf, "features": best_features},
                OUTPUTS_DIR / "rf_lightweight_model.joblib")
    print(f"\nSaved lightweight model (top-8 features: {best_features}) "
          f"-> outputs/rf_lightweight_model.joblib")
