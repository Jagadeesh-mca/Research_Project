"""
train_model.py
----------------
Random Forest classifier -> Benign/malicious prediction stage.
"""

import time
import time
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    precision_score, recall_score, accuracy_score, roc_auc_score,
    roc_curve, auc, precision_recall_curve, average_precision_score
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def train_random_forest(X_train, y_train, n_estimators=100, max_depth=16, seed=42):
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",
        n_jobs=-1,
        random_state=seed,
    )
    t0 = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - t0
    return clf, train_time


def evaluate(clf, X_test, y_test, label_encoder):
    t0 = time.time()
    y_pred = clf.predict(X_test)
    predict_time = time.time() - t0
    y_proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    throughput = len(X_test) / max(predict_time, 1e-6)

    roc_auc = float(roc_auc_score(y_test, y_proba)) if y_proba is not None else 0.0
    pr_auc = float(average_precision_score(y_test, y_proba)) if y_proba is not None else 0.0

    latency_per_flow_ms = float((predict_time / len(X_test)) * 1000.0)
    latency_per_1k_flows_ms = float(latency_per_flow_ms * 1000.0)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "false_positive_rate": float(fpr),
        "false_negative_rate": float(fnr),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "predict_time_s": float(predict_time),
        "latency_per_flow_ms": latency_per_flow_ms,
        "latency_per_1k_flows_ms": latency_per_1k_flows_ms,
        "predict_time_per_1k_ms": latency_per_1k_flows_ms,
        "throughput_flows_per_s": float(throughput),
    }
    report = classification_report(
        y_test, y_pred, target_names=label_encoder.classes_, output_dict=True
    )
    return {
        "metrics": metrics,
        "report": report,
        "confusion_matrix": cm,
        "y_pred": y_pred,
        "y_proba": y_proba,
    }


def plot_confusion_matrix(cm, class_names, out_path, show_plot=True):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 13, "weight": "bold"})
    plt.xlabel("Predicted Label", fontsize=11, fontweight="bold")
    plt.ylabel("True Label", fontsize=11, fontweight="bold")
    plt.title("Confusion Matrix - Random Forest IDS", fontsize=12, fontweight="bold", pad=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_roc_curve(y_true, y_proba, out_path, show_plot=True):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6.5, 5))
    plt.plot(fpr, tpr, color="#1f77b4", lw=2.5, label=f"Random Forest (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Random Chance")
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=11, fontweight="bold")
    plt.ylabel("True Positive Rate (Recall)", fontsize=11, fontweight="bold")
    plt.title("Receiver Operating Characteristic (ROC) Curve", fontsize=12, fontweight="bold", pad=10)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_precision_recall_curve(y_true, y_proba, out_path, show_plot=True):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)

    plt.figure(figsize=(6.5, 5))
    plt.plot(recall, precision, color="#2ca02c", lw=2.5, label=f"Random Forest (PR-AUC = {pr_auc:.4f})")
    plt.xlabel("Recall", fontsize=11, fontweight="bold")
    plt.ylabel("Precision", fontsize=11, fontweight="bold")
    plt.title("Precision-Recall (PR) Curve", fontsize=12, fontweight="bold", pad=10)
    plt.xlim([0.0, 1.02])
    plt.ylim([0.0, 1.05])
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    if show_plot:
        plt.show()
    else:
        plt.close()


def benchmark_baselines(X_train, y_train, X_test, y_test, label_encoder):
    """
    Benchmarks multiple classifiers under identical experimental conditions:
    - Logistic Regression (linear baseline)
    - Decision Tree (interpretable tree baseline)
    - Extra Trees (randomized ensemble baseline)
    - Random Forest (proposed ensemble model)
    """
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=16, class_weight="balanced", random_state=42),
        "Extra Trees": ExtraTreesClassifier(n_estimators=100, max_depth=16, class_weight="balanced", n_jobs=-1, random_state=42),
        "Random Forest (Proposed)": RandomForestClassifier(n_estimators=100, max_depth=16, class_weight="balanced", n_jobs=-1, random_state=42)
    }

    records = []
    for name, model in models.items():
        t0 = time.time()
        model.fit(X_train, y_train)
        t_train = time.time() - t0

        t1 = time.time()
        y_pred = model.predict(X_test)
        t_pred = time.time() - t1

        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
            roc = roc_auc_score(y_test, y_proba)
            pr_auc = average_precision_score(y_test, y_proba)
        else:
            roc = 0.0
            pr_auc = 0.0

        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        records.append({
            "Classifier": name,
            "Accuracy": float(accuracy_score(y_test, y_pred)),
            "Precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "Recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "F1-Score": float(f1_score(y_test, y_pred, zero_division=0)),
            "ROC-AUC": float(roc),
            "PR-AUC": float(pr_auc),
            "FPR": float(fpr),
            "FNR": float(fnr),
            "Train_Time_s": float(t_train),
            "Latency_per_Flow_ms": float((t_pred / len(X_test)) * 1000.0),
            "Latency_per_1k_Flows_ms": float((t_pred / len(X_test)) * 1000.0 * 1000.0),
            "Inference_Latency_ms": float((t_pred / len(X_test)) * 1000.0 * 1000.0),
            "Throughput_Flows_s": float(len(X_test) / max(t_pred, 1e-6))
        })

    return pd.DataFrame(records)


def plot_model_comparison(df_models: pd.DataFrame, out_path: str, show_plot=True):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # 1. F1-Score
    axes[0, 0].bar(df_models["Classifier"], df_models["F1-Score"], color="#1f77b4", edgecolor="black", alpha=0.85)
    axes[0, 0].set_title("F1-Score Comparison", fontsize=11, fontweight="bold")
    axes[0, 0].set_ylim([max(0.8, df_models["F1-Score"].min() * 0.95), 1.02])
    axes[0, 0].tick_params(axis="x", rotation=15)
    axes[0, 0].grid(axis="y", linestyle=":", alpha=0.6)

    # 2. ROC-AUC
    axes[0, 1].bar(df_models["Classifier"], df_models["ROC-AUC"], color="#2ca02c", edgecolor="black", alpha=0.85)
    axes[0, 1].set_title("ROC-AUC Comparison", fontsize=11, fontweight="bold")
    axes[0, 1].set_ylim([max(0.8, df_models["ROC-AUC"].min() * 0.95), 1.02])
    axes[0, 1].tick_params(axis="x", rotation=15)
    axes[0, 1].grid(axis="y", linestyle=":", alpha=0.6)

    # 3. Inference Latency
    axes[1, 0].bar(df_models["Classifier"], df_models["Inference_Latency_ms"], color="#ff7f0e", edgecolor="black", alpha=0.85)
    axes[1, 0].set_title("Inference Latency (ms / 1k flows - Lower is Better)", fontsize=11, fontweight="bold")
    axes[1, 0].tick_params(axis="x", rotation=15)
    axes[1, 0].grid(axis="y", linestyle=":", alpha=0.6)

    # 4. Throughput
    axes[1, 1].bar(df_models["Classifier"], df_models["Throughput_Flows_s"] / 1000.0, color="#d62728", edgecolor="black", alpha=0.85)
    axes[1, 1].set_title("Throughput (kFlows / sec - Higher is Better)", fontsize=11, fontweight="bold")
    axes[1, 1].tick_params(axis="x", rotation=15)
    axes[1, 1].grid(axis="y", linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    if show_plot:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    from preprocessing import clean, encode_and_split

    df = pd.read_csv(PROJECT_ROOT / "data" / "demo_traffic.csv")
    df = clean(df)
    split = encode_and_split(df)

    clf, train_time = train_random_forest(split["X_train"], split["y_train"])
    print(f"Trained in {train_time:.2f}s on {len(split['X_train']):,} rows, "
          f"{len(split['feature_cols'])} features")

    results = evaluate(clf, split["X_test"], split["y_test"], split["label_encoder"])
    print("\n--- Full-feature model performance ---")
    for k, v in results["metrics"].items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print("\nConfusion matrix [rows=true, cols=pred] (Benign, Malicious):")
    print(results["confusion_matrix"])

    joblib.dump(clf, OUTPUTS_DIR / "rf_full_model.joblib")
    joblib.dump(split, OUTPUTS_DIR / "split.joblib")
    print(f"\nSaved model -> {OUTPUTS_DIR / 'rf_full_model.joblib'}")
