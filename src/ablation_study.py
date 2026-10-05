"""
ablation_study.py
-----------------
Conducts a systematic ablation study evaluating the individual and cumulative impact
of the architectural components proposed in the paper and IDS framework:
1. Config A1: Raw baseline (unscaled features, 68 features, unweighted RF)
2. Config A2: + Min-Max Normalization (Paper Sec IV.3)
3. Config A3: + Multicollinearity Reduction (|r| > 0.90)
4. Config A4: + Balanced Class Weighting (Full Proposed Model)
5. Config A5: SHAP-Distilled Lightweight Model (Top-8 Features)
"""

import time
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from correlation_analysis import drop_correlated_features


def run_ablation_experiments(split, ranking, top_k=8, seed=42):
    results = []

    X_tr_raw = split["X_train_raw"]
    X_te_raw = split["X_test_raw"]
    y_tr = split["y_train"]
    y_te = split["y_test"]
    all_features = list(X_tr_raw.columns)

    # --- Config A1: Raw Baseline (Unscaled, All Features, Unweighted RF) ---
    t0 = time.time()
    rf_a1 = RandomForestClassifier(n_estimators=100, max_depth=16, random_state=seed, n_jobs=-1)
    rf_a1.fit(X_tr_raw, y_tr)
    t_tr_a1 = time.time() - t0

    t1 = time.time()
    p_a1 = rf_a1.predict(X_te_raw)
    t_pred_a1 = time.time() - t1
    prob_a1 = rf_a1.predict_proba(X_te_raw)[:, 1]
    cm_a1 = confusion_matrix(y_te, p_a1)
    tn, fp, fn, tp = cm_a1.ravel()

    results.append({
        "Configuration": "A1: Raw Unscaled (All Feats, Unweighted)",
        "Features": X_tr_raw.shape[1],
        "Accuracy": float(accuracy_score(y_te, p_a1)),
        "Precision": float(precision_score(y_te, p_a1, zero_division=0)),
        "Recall": float(recall_score(y_te, p_a1, zero_division=0)),
        "F1-Score": float(f1_score(y_te, p_a1, zero_division=0)),
        "ROC-AUC": float(roc_auc_score(y_te, prob_a1)),
        "PR-AUC": float(average_precision_score(y_te, prob_a1)),
        "FPR": float(fp / (fp + tn) if (fp + tn) > 0 else 0),
        "FNR": float(fn / (fn + tp) if (fn + tp) > 0 else 0),
        "Train_Time_s": float(t_tr_a1),
        "Latency_ms_per_flow": float((t_pred_a1 / len(X_te_raw)) * 1000.0),
        "Latency_ms_1k": float((t_pred_a1 / len(X_te_raw)) * 1000.0 * 1000.0),
        "Throughput_Flows_s": float(len(X_te_raw) / max(t_pred_a1, 1e-6)),
    })

    # --- Config A2: + Min-Max Normalization (Paper Sec IV.3) ---
    scaler = MinMaxScaler()
    X_tr_s = pd.DataFrame(scaler.fit_transform(X_tr_raw), columns=all_features)
    X_te_s = pd.DataFrame(scaler.transform(X_te_raw), columns=all_features)

    t0 = time.time()
    rf_a2 = RandomForestClassifier(n_estimators=100, max_depth=16, random_state=seed, n_jobs=-1)
    rf_a2.fit(X_tr_s, y_tr)
    t_tr_a2 = time.time() - t0

    t1 = time.time()
    p_a2 = rf_a2.predict(X_te_s)
    t_pred_a2 = time.time() - t1
    prob_a2 = rf_a2.predict_proba(X_te_s)[:, 1]
    cm_a2 = confusion_matrix(y_te, p_a2)
    tn, fp, fn, tp = cm_a2.ravel()

    results.append({
        "Configuration": "A2: + Min-Max Normalization",
        "Features": X_tr_s.shape[1],
        "Accuracy": float(accuracy_score(y_te, p_a2)),
        "Precision": float(precision_score(y_te, p_a2, zero_division=0)),
        "Recall": float(recall_score(y_te, p_a2, zero_division=0)),
        "F1-Score": float(f1_score(y_te, p_a2, zero_division=0)),
        "ROC-AUC": float(roc_auc_score(y_te, prob_a2)),
        "PR-AUC": float(average_precision_score(y_te, prob_a2)),
        "FPR": float(fp / (fp + tn) if (fp + tn) > 0 else 0),
        "FNR": float(fn / (fn + tp) if (fn + tp) > 0 else 0),
        "Train_Time_s": float(t_tr_a2),
        "Latency_ms_per_flow": float((t_pred_a2 / len(X_te_s)) * 1000.0),
        "Latency_ms_1k": float((t_pred_a2 / len(X_te_s)) * 1000.0 * 1000.0),
        "Throughput_Flows_s": float(len(X_te_s) / max(t_pred_a2, 1e-6)),
    })

    # --- Config A3: + Multicollinearity Reduction (|r| > 0.90) ---
    X_tr_r, X_te_r, kept_feats, dropped_feats, _ = drop_correlated_features(
        X_tr_s, X_te_s, all_features, threshold=0.90
    )

    t0 = time.time()
    rf_a3 = RandomForestClassifier(n_estimators=100, max_depth=16, random_state=seed, n_jobs=-1)
    rf_a3.fit(X_tr_r, y_tr)
    t_tr_a3 = time.time() - t0

    t1 = time.time()
    p_a3 = rf_a3.predict(X_te_r)
    t_pred_a3 = time.time() - t1
    prob_a3 = rf_a3.predict_proba(X_te_r)[:, 1]
    cm_a3 = confusion_matrix(y_te, p_a3)
    tn, fp, fn, tp = cm_a3.ravel()

    results.append({
        "Configuration": "A3: + Correlation Filter (|r| > 0.90)",
        "Features": X_tr_r.shape[1],
        "Accuracy": float(accuracy_score(y_te, p_a3)),
        "Precision": float(precision_score(y_te, p_a3, zero_division=0)),
        "Recall": float(recall_score(y_te, p_a3, zero_division=0)),
        "F1-Score": float(f1_score(y_te, p_a3, zero_division=0)),
        "ROC-AUC": float(roc_auc_score(y_te, prob_a3)),
        "PR-AUC": float(average_precision_score(y_te, prob_a3)),
        "FPR": float(fp / (fp + tn) if (fp + tn) > 0 else 0),
        "FNR": float(fn / (fn + tp) if (fn + tp) > 0 else 0),
        "Train_Time_s": float(t_tr_a3),
        "Latency_ms_per_flow": float((t_pred_a3 / len(X_te_r)) * 1000.0),
        "Latency_ms_1k": float((t_pred_a3 / len(X_te_r)) * 1000.0 * 1000.0),
        "Throughput_Flows_s": float(len(X_te_r) / max(t_pred_a3, 1e-6)),
    })

    # --- Config A4: + Balanced Class Weighting (Proposed Full) ---
    t0 = time.time()
    rf_a4 = RandomForestClassifier(n_estimators=100, max_depth=16, class_weight="balanced", random_state=seed, n_jobs=-1)
    rf_a4.fit(X_tr_r, y_tr)
    t_tr_a4 = time.time() - t0

    t1 = time.time()
    p_a4 = rf_a4.predict(X_te_r)
    t_pred_a4 = time.time() - t1
    prob_a4 = rf_a4.predict_proba(X_te_r)[:, 1]
    cm_a4 = confusion_matrix(y_te, p_a4)
    tn, fp, fn, tp = cm_a4.ravel()

    results.append({
        "Configuration": "A4: + Balanced Class Weight (Proposed Full)",
        "Features": X_tr_r.shape[1],
        "Accuracy": float(accuracy_score(y_te, p_a4)),
        "Precision": float(precision_score(y_te, p_a4, zero_division=0)),
        "Recall": float(recall_score(y_te, p_a4, zero_division=0)),
        "F1-Score": float(f1_score(y_te, p_a4, zero_division=0)),
        "ROC-AUC": float(roc_auc_score(y_te, prob_a4)),
        "PR-AUC": float(average_precision_score(y_te, prob_a4)),
        "FPR": float(fp / (fp + tn) if (fp + tn) > 0 else 0),
        "FNR": float(fn / (fn + tp) if (fn + tp) > 0 else 0),
        "Train_Time_s": float(t_tr_a4),
        "Latency_ms_per_flow": float((t_pred_a4 / len(X_te_r)) * 1000.0),
        "Latency_ms_1k": float((t_pred_a4 / len(X_te_r)) * 1000.0 * 1000.0),
        "Throughput_Flows_s": float(len(X_te_r) / max(t_pred_a4, 1e-6)),
    })

    # --- Config A5: SHAP-Distilled Lightweight Model (Top-8 Features) ---
    top_features = ranking.head(top_k).index.tolist()
    X_tr_lw = X_tr_s[top_features]
    X_te_lw = X_te_s[top_features]

    t0 = time.time()
    rf_a5 = RandomForestClassifier(n_estimators=100, max_depth=16, class_weight="balanced", random_state=seed, n_jobs=-1)
    rf_a5.fit(X_tr_lw, y_tr)
    t_tr_a5 = time.time() - t0

    t1 = time.time()
    p_a5 = rf_a5.predict(X_te_lw)
    t_pred_a5 = time.time() - t1
    prob_a5 = rf_a5.predict_proba(X_te_lw)[:, 1]
    cm_a5 = confusion_matrix(y_te, p_a5)
    tn, fp, fn, tp = cm_a5.ravel()

    results.append({
        "Configuration": f"A5: SHAP Lightweight (Top-{top_k} Feats)",
        "Features": top_k,
        "Accuracy": float(accuracy_score(y_te, p_a5)),
        "Precision": float(precision_score(y_te, p_a5, zero_division=0)),
        "Recall": float(recall_score(y_te, p_a5, zero_division=0)),
        "F1-Score": float(f1_score(y_te, p_a5, zero_division=0)),
        "ROC-AUC": float(roc_auc_score(y_te, prob_a5)),
        "PR-AUC": float(average_precision_score(y_te, prob_a5)),
        "FPR": float(fp / (fp + tn) if (fp + tn) > 0 else 0),
        "FNR": float(fn / (fn + tp) if (fn + tp) > 0 else 0),
        "Train_Time_s": float(t_tr_a5),
        "Latency_ms_per_flow": float((t_pred_a5 / len(X_te_lw)) * 1000.0),
        "Latency_ms_1k": float((t_pred_a5 / len(X_te_lw)) * 1000.0 * 1000.0),
        "Throughput_Flows_s": float(len(X_te_lw) / max(t_pred_a5, 1e-6)),
    })

    return pd.DataFrame(results)


def plot_ablation_study(df_ablation: pd.DataFrame, out_path: str, show_plot=True):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    labels = [c.split(":")[0] for c in df_ablation["Configuration"]]

    # 1. F1-Score
    axes[0, 0].bar(labels, df_ablation["F1-Score"], color="#1f77b4", edgecolor="black", alpha=0.85)
    axes[0, 0].set_title("Detection F1-Score Across Configurations", fontsize=11, fontweight="bold")
    axes[0, 0].set_ylabel("F1-Score", fontsize=11, fontweight="bold")
    axes[0, 0].set_ylim([max(0.8, df_ablation["F1-Score"].min() * 0.95), 1.02])
    axes[0, 0].grid(axis="y", linestyle=":", alpha=0.6)

    # 2. Number of Features
    axes[0, 1].bar(labels, df_ablation["Features"], color="#9467bd", edgecolor="black", alpha=0.85)
    axes[0, 1].set_title("Feature Dimensionality Reduction", fontsize=11, fontweight="bold")
    axes[0, 1].set_ylabel("Number of Features", fontsize=11, fontweight="bold")
    axes[0, 1].grid(axis="y", linestyle=":", alpha=0.6)

    # 3. Latency
    axes[1, 0].bar(labels, df_ablation["Latency_ms_1k"], color="#ff7f0e", edgecolor="black", alpha=0.85)
    axes[1, 0].set_title("Inference Latency (ms / 1k flows - Lower is Better)", fontsize=11, fontweight="bold")
    axes[1, 0].set_ylabel("Latency (ms / 1k flows)", fontsize=11, fontweight="bold")
    axes[1, 0].grid(axis="y", linestyle=":", alpha=0.6)

    # 4. Throughput
    axes[1, 1].bar(labels, df_ablation["Throughput_Flows_s"] / 1000.0, color="#2ca02c", edgecolor="black", alpha=0.85)
    axes[1, 1].set_title("Detection Throughput (kFlows / sec - Higher is Better)", fontsize=11, fontweight="bold")
    axes[1, 1].set_ylabel("Throughput (kFlows / sec)", fontsize=11, fontweight="bold")
    axes[1, 1].grid(axis="y", linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    if show_plot:
        plt.show()
    else:
        plt.close()
