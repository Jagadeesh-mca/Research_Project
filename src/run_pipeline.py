"""
run_pipeline.py
------------------
Explainable Machine Learning Intrusion Detection System (CSE-CIC-IDS2018).

Features:
1. Prompts for user input FIRST before executing.
2. Dynamically calculates results fresh from current execution (no cached results).
3. Directly opens and displays Matplotlib/SHAP visualizations on the spot using plt.show().
4. Every new run clears previous visualizations and displays the fresh result for the latest input.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

OUT = PROJECT_ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
FIG_DIR = OUT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

from generate_demo_data import generate, load_real_dataset
from preprocessing import clean, encode_and_split, verify_data_leakage
from train_model import (
    train_random_forest, evaluate, benchmark_baselines,
    plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve,
    plot_model_comparison
)
from correlation_analysis import drop_correlated_features, plot_correlation_heatmap
from shap_explain import (
    build_explainer, compute_shap_values, global_feature_importance,
    plot_summary, plot_bar, plot_dependence, plot_waterfall, explain_single_prediction
)
from false_positive_analysis import find_false_positives, analyse_false_positives, plot_fp_drivers
from feature_reduction import compare_full_vs_reduced, plot_comparison
from stability_analysis import (
    neighbour_explanation_consistency, bootstrap_ranking_stability,
    summarise_bootstrap_stability, plot_stability
)
from ablation_study import run_ablation_experiments, plot_ablation_study
from report_generator import generate_html_report, generate_markdown_report

REAL_DATA_FILE = PROJECT_ROOT / "data" / "02-14-2018.csv"
REAL_DATA_PATHS = [str(REAL_DATA_FILE)] if REAL_DATA_FILE.exists() else []


def safe_show(show_plot=True):
    """Displays the active Matplotlib figure directly on screen in VS Code if show_plot is True."""
    try:
        if show_plot:
            plt.show()
    except Exception as e:
        print(f"  [Display notice: {e}]")
    finally:
        plt.close("all")


def configure_plot_backend(show_gui):
    if not show_gui:
        plt.switch_backend("Agg")


def plot_dataset_distribution(df: pd.DataFrame, out_path: str, show_plot=True):
    plt.close("all")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    label_counts = df["Label"].value_counts()
    colors = ["#1f77b4", "#d62728"]
    axes[0].pie(label_counts, labels=label_counts.index, autopct="%1.1f%%",
                colors=colors, startangle=140, explode=(0, 0.08),
                wedgeprops={"edgecolor": "black", "linewidth": 1.2})
    axes[0].set_title("Current Run: Class Distribution", fontsize=11, fontweight="bold")

    cat_counts = df["Attack_Category"].value_counts()
    cat_colors = ["#2ca02c" if c == "Benign" else "#ff7f0e" for c in cat_counts.index]
    axes[1].bar(cat_counts.index, cat_counts.values, color=cat_colors, edgecolor="black", alpha=0.85)
    for i, v in enumerate(cat_counts.values):
        axes[1].text(i, v + max(cat_counts.values)*0.015, f"{v:,}", ha="center", fontsize=9, fontweight="bold")
    axes[1].set_title("Current Run: Attack Category Breakdown", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Flow Count", fontsize=10, fontweight="bold")
    axes[1].grid(axis="y", linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    if show_plot:
        print("  -> Displaying Class Distribution plot on the spot (close window to proceed)...")
    safe_show(show_plot)


def run_full_pipeline(sample_size=100_000, seed=42, show_gui=True):
    configure_plot_backend(show_gui)
    plt.close("all")
    print("\n" + "=" * 60)
    print(f"EXECUTING FRESH IDS RESEARCH PIPELINE (N={sample_size:,}, SEED={seed})")
    print("=" * 60)
    pipeline_t0 = time.time()
    summary = {"pipeline_stage_log": []}

    # Stage 1: Load Data
    print("\n[1/11] Loading dataset...")
    t0 = time.time()
    if REAL_DATA_PATHS and Path(REAL_DATA_PATHS[0]).exists():
        df_raw = load_real_dataset(REAL_DATA_PATHS)
        source_note = f"Real CSE-CIC-IDS2018 cloud dataset ({REAL_DATA_PATHS[0]})"
    else:
        df_raw = generate()
        source_note = "Synthetic proof-of-concept dataset replicating CSE-CIC-IDS2018 schema."
    load_time = time.time() - t0
    print(f"  Loaded {len(df_raw):,} raw flows in {load_time:.2f}s.")

    # Stage 2: Clean & De-duplicate
    print("\n[2/11] Preprocessing: De-duplication, Zero-variance removal, Scaling & Leakage Checks...")
    t0 = time.time()
    df_clean, dropped_const_cols = clean(df_raw, drop_duplicates=True, drop_zero_variance=True)
    if sample_size and len(df_clean) > sample_size:
        df = df_clean.sample(n=sample_size, random_state=seed).reset_index(drop=True)
        source_note += f" (sampled {sample_size:,} flows)"
    else:
        df = df_clean.copy()

    split = encode_and_split(df, test_size=0.20, seed=seed, apply_minmax=True)
    prep_time = time.time() - t0
    leakage = split["leakage_report"]
    print(f"  Clean unique flows: {len(df_clean):,} (using {len(df):,})")
    print(f"  Leakage verification -> Overlap: {leakage['train_test_overlap_count']} rows, Leakage-free: {leakage['leakage_free']}")
    plot_dataset_distribution(df, str(FIG_DIR / "class_distribution.png"), show_plot=show_gui)

    summary["dataset"] = {
        "n_rows": len(df),
        "n_benign": int((df["Label"] == "Benign").sum()),
        "n_malicious": int((df["Label"] == "Malicious").sum()),
        "attack_category_counts": df["Attack_Category"].value_counts().to_dict(),
        "note": source_note,
        "load_time_s": round(load_time, 2),
    }
    summary["preprocessing"] = {
        "preprocessing_time_s": round(prep_time, 2),
        "zero_variance_columns_dropped": dropped_const_cols,
        "n_features_initial": len(split["feature_cols"]) + len(dropped_const_cols),
        "n_features_after_cleaning": len(split["feature_cols"]),
        "train_size": len(split["X_train"]),
        "test_size": len(split["X_test"]),
    }
    summary["leakage_checks"] = leakage

    # Stage 3: Correlation & Multicollinearity
    print("\n[3/11] Correlation & Multicollinearity Filtering (|r| > 0.90 on X_train only)...")
    t0 = time.time()
    X_train_r, X_test_r, kept_features, dropped_features, corr = drop_correlated_features(
        split["X_train"], split["X_test"], split["feature_cols"], threshold=0.90
    )
    corr_time = time.time() - t0
    print(f"  Pruned {len(dropped_features)} multicollinear features. Retained: {len(kept_features)} features.")
    if show_gui:
        print("  -> Displaying Correlation Heatmap on the spot (close window to proceed)...")
    plot_correlation_heatmap(corr, str(FIG_DIR / "correlation_heatmap.png"), dropped_features, show_plot=show_gui)

    split["X_train"], split["X_test"], split["feature_cols"] = X_train_r, X_test_r, kept_features
    summary["correlation_analysis"] = {
        "correlation_filter_time_s": round(corr_time, 2),
        "threshold": 0.90,
        "n_features_dropped": len(dropped_features),
        "dropped_features": dropped_features,
        "n_features_remaining": len(kept_features),
    }

    # Stage 4: Multi-Model Baseline Benchmarking
    print("\n[4/11] Benchmarking multi-model baselines (LR, DT, Extra Trees, Random Forest)...")
    baseline_df = benchmark_baselines(split["X_train"], split["y_train"], split["X_test"], split["y_test"], split["label_encoder"])
    baseline_df.to_csv(str(OUT / "baseline_comparison.csv"), index=False)
    summary["baseline_benchmarks"] = baseline_df.round(4).to_dict(orient="records")
    print(baseline_df[["Classifier", "Accuracy", "F1-Score", "ROC-AUC", "Latency_per_Flow_ms", "Latency_per_1k_Flows_ms", "Throughput_Flows_s"]].to_string(index=False))
    if show_gui:
        print("  -> Displaying Model Comparison benchmark plot on the spot (close window to proceed)...")
    plot_model_comparison(baseline_df, str(FIG_DIR / "model_comparison.png"), show_plot=show_gui)

    # Stage 5: Train Proposed Full Model
    print("\n[5/11] Training Proposed Full Random Forest Model...")
    clf, train_time = train_random_forest(split["X_train"], split["y_train"], n_estimators=100, max_depth=16, seed=seed)
    results = evaluate(clf, split["X_test"], split["y_test"], split["label_encoder"])
    print(f"  Accuracy: {results['metrics']['accuracy']*100:.2f}% | F1: {results['metrics']['f1']:.4f} | ROC-AUC: {results['metrics']['roc_auc']:.4f}")
    print(f"  Inference Latency: {results['metrics']['latency_per_flow_ms']:.6f} ms/flow ({results['metrics']['latency_per_1k_flows_ms']:.4f} ms/1k) | Throughput: {results['metrics']['throughput_flows_per_s']:,.0f} flows/s")
    print(f"  Confusion Matrix: {results['confusion_matrix'].tolist()}")
    if show_gui:
        print("  -> Displaying Confusion Matrix on the spot...")
    plot_confusion_matrix(results["confusion_matrix"], split["label_encoder"].classes_, str(FIG_DIR / "confusion_matrix.png"), show_plot=show_gui)
    plot_roc_curve(split["y_test"], results["y_proba"], str(FIG_DIR / "roc_curve.png"), show_plot=show_gui)
    plot_precision_recall_curve(split["y_test"], results["y_proba"], str(FIG_DIR / "pr_curve.png"), show_plot=show_gui)

    summary["full_model"] = {
        "n_features": len(split["feature_cols"]),
        "train_time_s": round(train_time, 4),
        **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in results["metrics"].items()},
        "confusion_matrix": results["confusion_matrix"].tolist(),
        "per_class_report": results["report"],
    }
    joblib.dump(clf, str(OUT / "rf_full_model.joblib"))
    joblib.dump(split, str(OUT / "split.joblib"))

    # Stage 6: SHAP Explainability (Leakage-Free Ranking on X_train)
    print("\n[6/11] Computing SHAP explainability via TreeExplainer...")
    t0 = time.time()
    explainer = build_explainer(clf)
    
    # 1. Feature selection ranking: computed STRICTLY on X_train (Leakage-Free)
    shap_values_train, X_sample_train = compute_shap_values(explainer, split["X_train"], sample_size=1000, seed=seed)
    ranking = global_feature_importance(shap_values_train, split["feature_cols"])
    shap_time = time.time() - t0
    ranking.to_csv(str(OUT / "shap_feature_ranking.csv"))
    print(f"  SHAP feature ranking computed on X_train in {shap_time:.2f}s. Top 3 drivers: {list(ranking.index[:3])}")

    # 2. Local test explanations and beeswarm: computed on X_test for triage evaluation
    shap_values_test, X_sample_test = compute_shap_values(explainer, split["X_test"], sample_size=1000, seed=seed)
    if show_gui:
        print("  -> Displaying SHAP Summary Beeswarm on the spot...")
    plot_summary(shap_values_test, X_sample_test, str(FIG_DIR / "shap_summary.png"), show_plot=show_gui)
    plot_bar(ranking, str(FIG_DIR / "shap_bar.png"), top_n=15, show_plot=show_gui)
    plot_dependence(shap_values_test, X_sample_test, ranking.index[0], str(FIG_DIR / "shap_dependence.png"), show_plot=show_gui)

    # Local Explanations
    malicious_idx = split["X_test"][split["y_test"] == 1].index
    proba_mal = clf.predict_proba(split["X_test"].loc[malicious_idx])[:, 1]
    best_mal_idx = malicious_idx[np.argmax(proba_mal)]
    row_mal = split["X_test"].loc[[best_mal_idx]]
    if show_gui:
        print(f"  -> Displaying Local Malicious Waterfall ({split['cat_test'].loc[best_mal_idx]})...")
    plot_waterfall(explainer, row_mal, split["feature_cols"], str(FIG_DIR / "shap_waterfall_malicious.png"),
                   title=f"Local SHAP Explanation: Confirmed Malicious Attack ({split['cat_test'].loc[best_mal_idx]})", show_plot=show_gui)

    summary["shap"] = {
        "shap_computation_time_s": round(shap_time, 2),
        "top_15_features": ranking.head(15).round(4).to_dict(),
    }
    joblib.dump({"explainer": explainer, "ranking": ranking}, str(OUT / "shap_artifacts.joblib"))

    # Stage 7: False Positive Analysis
    print("\n[7/11] Diagnosing false positives and near-boundary risk...")
    X_fp, X_fn, fp_mask, fn_mask = find_false_positives(clf, split["X_test"], split["y_test"])
    X_benign = split["X_test"][split["y_test"] == 0]
    driving_features, _, is_near_boundary = analyse_false_positives(
        clf, explainer, X_fp, split["feature_cols"], X_benign=X_benign, top_n=8
    )
    if driving_features is not None:
        if show_gui:
            print("  -> Displaying False Positive / Risk Drivers plot on the spot...")
        plot_fp_drivers(driving_features, str(FIG_DIR / "false_positive_drivers.png"), is_near_boundary, show_plot=show_gui)

    summary["false_positive_analysis"] = {
        "n_false_positives": int(len(X_fp)),
        "n_false_negatives": int(len(X_fn)),
        "fp_rate_pct": round(len(X_fp) / len(split["X_test"]) * 100, 6),
        "fn_rate_pct": round(len(X_fn) / len(split["X_test"]) * 100, 6),
        "is_near_boundary_analysis": is_near_boundary,
        "top_driving_features": driving_features.round(4).to_dict() if driving_features is not None else {},
    }

    # Stage 8: Feature Reduction
    print("\n[8/11] Evaluating SHAP-driven feature distillation (Full vs Top-5, 8, 10, 12, 15)...")
    comparison_df, models = compare_full_vs_reduced(split, ranking, k_values=(5, 8, 10, 12, 15))
    print(comparison_df[["model", "n_features", "accuracy", "f1", "latency_per_flow_ms", "latency_per_1k_ms", "throughput_flows_s"]].to_string(index=False))
    if show_gui:
        print("  -> Displaying Feature Reduction Trade-off on the spot...")
    plot_comparison(comparison_df, str(FIG_DIR / "feature_reduction_tradeoff.png"), show_plot=show_gui)

    best_k = "top_8" if "top_8" in models else list(models.keys())[1]
    best_clf, best_res, best_features = models[best_k]
    joblib.dump({"model": best_clf, "features": best_features}, str(OUT / "rf_lightweight_model.joblib"))
    summary["feature_reduction"] = comparison_df.round(4).to_dict(orient="records")
    summary["lightweight_model_features"] = best_features

    # Stage 9: Explanation Stability
    print("\n[9/11] Assessing explanation stability (local neighbour consistency & bootstrap)...")
    spearmans, cosines = neighbour_explanation_consistency(explainer, split["X_test"], n_samples=100, seed=seed)
    rankings = bootstrap_ranking_stability(
        split["X_train"], split["y_train"], split["feature_cols"],
        n_bootstraps=3, sample_size=1000, seed=seed, n_estimators=60,
    )
    boot_summary = summarise_bootstrap_stability(rankings, top_k=8)
    if show_gui:
        print("  -> Displaying Explanation Stability distributions on the spot...")
    plot_stability(spearmans, cosines, str(FIG_DIR / "stability_analysis.png"), show_plot=show_gui)
    summary["stability_analysis"] = {
        "neighbour_consistency": {
            "n_pairs": int(len(spearmans)),
            "median_spearman": round(float(np.median(spearmans)), 4) if len(spearmans) else 1.0,
            "mean_spearman": round(float(np.mean(spearmans)), 4) if len(spearmans) else 1.0,
            "median_cosine": round(float(np.median(cosines)), 4) if len(cosines) else 1.0,
            "mean_cosine": round(float(np.mean(cosines)), 4) if len(cosines) else 1.0,
            "pct_spearman_above_0_7": round(float((spearmans > 0.7).mean() * 100), 2) if len(spearmans) else 100.0,
        },
        "bootstrap_ranking_stability": {k: round(v, 4) for k, v in boot_summary.items()},
    }

    # Stage 10: Systematic Ablation Study
    print("\n[10/11] Conducting systematic architecture ablation study (5 configurations)...")
    ablation_df = run_ablation_experiments(split, ranking, top_k=8, seed=seed)
    print(ablation_df[["Configuration", "Features", "F1-Score", "Latency_ms_per_flow", "Latency_ms_1k", "Throughput_Flows_s"]].to_string(index=False))
    if show_gui:
        print("  -> Displaying Ablation Study evaluation on the spot...")
    plot_ablation_study(ablation_df, str(FIG_DIR / "ablation_study.png"), show_plot=show_gui)
    summary["ablation_study"] = ablation_df.round(4).to_dict(orient="records")

    # Stage 11: Export Reports
    print("\n[11/11] Exporting comprehensive results summary & reports...")
    total_elapsed = round(time.time() - pipeline_t0, 2)
    summary["pipeline_execution_time_s"] = total_elapsed

    with open(str(OUT / "results_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    generate_html_report(summary, str(OUT / "research_report.html"))
    generate_markdown_report(summary, str(OUT / "research_report.md"))

    print("\n" + "=" * 50)
    print("IDS RESEARCH PIPELINE COMPLETE")
    print("=" * 50)
    print()
    print("Figures:")
    print("outputs/figures/")
    print()
    print("Research Report:")
    print("outputs/research_report.html")
    print("outputs/research_report.md")
    print()
    print("Results:")
    print("outputs/results_summary.json")
    print("=" * 50)


def run_live_flow_triage(show_gui=True):
    """Interactive mode: user selects/enters an input flow to classify and explain dynamically."""
    configure_plot_backend(show_gui)
    plt.close("all")
    print("\n" + "=" * 60)
    print("LIVE FLOW TRIAGE: DYNAMIC INSPECTION & SHAP EXPLANATION")
    print("=" * 60)
    print("Loading clean dataset and training model fresh in memory...")

    df_raw = load_real_dataset(REAL_DATA_PATHS) if (REAL_DATA_PATHS and Path(REAL_DATA_PATHS[0]).exists()) else generate()
    df_clean, _ = clean(df_raw, drop_duplicates=True, drop_zero_variance=True)
    df = df_clean.sample(n=min(30_000, len(df_clean)), random_state=42).reset_index(drop=True)
    split = encode_and_split(df, test_size=0.20, seed=42, apply_minmax=True)

    X_train_r, X_test_r, kept_features, _, _ = drop_correlated_features(split["X_train"], split["X_test"], split["feature_cols"], threshold=0.90)
    split["X_train"], split["X_test"], split["feature_cols"] = X_train_r, X_test_r, kept_features

    clf, _ = train_random_forest(split["X_train"], split["y_train"], n_estimators=60, max_depth=14)
    explainer = build_explainer(clf)

    while True:
        print("\nAvailable Input Options:")
        print("  [1] High-Confidence Malicious Flow (SSH-Bruteforce)")
        print("  [2] High-Confidence Malicious Flow (FTP-BruteForce)")
        print("  [3] Verified Benign Flow")
        print("  [4] Near-Boundary / High-Risk Benign Flow")
        print("  [5] Enter custom test sample index (0 to", len(split["X_test"])-1, ")")
        print("  [Q] Exit to menu")

        try:
            choice = input("\nEnter your selection [1-5 or Q] (default: 1): ").strip().lower()
            if not choice:
                choice = "1"
        except (EOFError, KeyboardInterrupt):
            break

        if choice == "q":
            print("Exiting Live Triage.")
            break

        row_idx = None
        target_name = "Custom Sample"

        if choice == "1":
            ssh_indices = split["X_test"][split["cat_test"] == "SSH-Bruteforce"].index
            if len(ssh_indices) > 0:
                row_idx = ssh_indices[0]
                target_name = "SSH-Bruteforce Flow"
            else:
                row_idx = split["X_test"][split["y_test"] == 1].index[0]
                target_name = "Malicious Flow"
        elif choice == "2":
            ftp_indices = split["X_test"][split["cat_test"] == "FTP-BruteForce"].index
            if len(ftp_indices) > 0:
                row_idx = ftp_indices[0]
                target_name = "FTP-BruteForce Flow"
            else:
                row_idx = split["X_test"][split["y_test"] == 1].index[0]
                target_name = "Malicious Flow"
        elif choice == "3":
            ben_indices = split["X_test"][split["y_test"] == 0].index
            row_idx = ben_indices[0]
            target_name = "Benign Flow"
        elif choice == "4":
            X_ben = split["X_test"][split["y_test"] == 0]
            p_attack = clf.predict_proba(X_ben)[:, 1]
            highest_risk = np.argmax(p_attack)
            row_idx = X_ben.index[highest_risk]
            target_name = "Highest-Risk Benign Flow"
        elif choice == "5":
            try:
                idx_str = input(f"Enter index [0 - {len(split['X_test'])-1}]: ").strip()
                row_idx = int(idx_str)
                target_name = f"Test Flow #{row_idx}"
            except (ValueError, EOFError):
                row_idx = 0
                target_name = "Test Flow #0"
        else:
            row_idx = split["X_test"].index[0]
            target_name = "Default Sample"

        # Dynamically predict & explain chosen flow
        X_row = split["X_test"].loc[[row_idx]]
        true_label = split["y_test"].loc[row_idx] if hasattr(split["y_test"], "loc") else split["y_test"][row_idx]
        true_cat = split["cat_test"].loc[row_idx] if hasattr(split["cat_test"], "loc") else str(true_label)

        exp_result = explain_single_prediction(clf, explainer, X_row, split["feature_cols"], split["label_encoder"])

        print("\n" + "-" * 55)
        print(f"=== LIVE ALERT TRIAGE: {target_name} ===")
        print("-" * 55)
        print(f"Sample Index:             {row_idx}")
        print(f"Ground Truth:             {true_cat}")
        print(f"Model Prediction:         {exp_result['prediction'].upper()}")
        print(f"Prediction Confidence:    {exp_result['confidence']*100:.2f}%")
        print("\nTop Contributing Features (SHAP values):")
        for feat, val in exp_result["top_features"].items():
            direction = "Attack (+)" if val > 0 else "Benign (-)"
            print(f"  * {feat:<25}: {val:+.4f} ({direction})")

        print(f"\nSOC Analyst Summary:\n{exp_result['explanation']}")
        if show_gui:
            print("\n-> Generating and displaying live SHAP waterfall visualization on the spot...")
        else:
            print("\n-> Generating live SHAP waterfall visualization...")

        # Clear any previous figures before rendering fresh one
        plt.close("all")
        plot_waterfall(
            explainer, X_row, split["feature_cols"],
            str(FIG_DIR / "live_triage_waterfall.png"),
            title=f"Dynamic SHAP Waterfall: {target_name} (Pred: {exp_result['prediction']}, Conf: {exp_result['confidence']*100:.1f}%)",
            show_plot=show_gui
        )

        try:
            again = input("\nWould you like to inspect another flow? [Y/n]: ").strip().lower()
            if again == "n":
                print("Finished live inspection.")
                break
        except (EOFError, KeyboardInterrupt):
            break


def main():
    parser = argparse.ArgumentParser(description="Explainable Intrusion Detection System")
    parser.add_argument("--mode", choices=["1", "2", "3"], help="Execution mode (1: Live Triage, 2: Full Pipeline, 3: Feature Reduction)")
    parser.add_argument("--sample-size", type=int, default=50_000, help="Sample size for training")
    parser.add_argument("--no-gui", action="store_true", help="Run without popping up interactive GUI windows")
    args, unknown = parser.parse_known_args()

    show_gui = not args.no_gui

    # If CLI arguments passed explicitly, execute directly
    if args.mode == "1":
        run_live_flow_triage(show_gui=show_gui)
        return
    elif args.mode == "2":
        run_full_pipeline(sample_size=args.sample_size, show_gui=show_gui)
        return
    elif args.mode == "3":
        print(f"\nRunning Feature Reduction study...")
        run_full_pipeline(sample_size=args.sample_size, show_gui=show_gui)
        return

    # Interactive Menu Prompt FIRST
    print("=" * 60)
    print("EXPLAINABLE INTRUSION DETECTION SYSTEM (CSE-CIC-IDS2018)")
    print("=" * 60)
    print("Select Execution Mode:")
    print("  [1] Live Flow Triage (Select/Input a flow -> instant prediction & SHAP waterfall)")
    print("  [2] Full Research Pipeline (Fresh calculation across all 11 stages with live figures)")
    print("  [3] Feature Reduction & Distillation Study (Custom Top-K evaluation)")

    try:
        choice = input("\nEnter choice [1, 2, or 3] (default: 1): ").strip()
        if not choice:
            choice = "1"
    except (EOFError, KeyboardInterrupt):
        choice = "1"

    if choice == "1":
        run_live_flow_triage(show_gui=show_gui)
    elif choice == "2":
        try:
            s_input = input("Enter sample size [default 50,000 for fast interactive / 100,000 for full]: ").strip()
            sample_size = int(s_input.replace(",", "")) if s_input else 50_000
        except (ValueError, EOFError):
            sample_size = 50_000
        run_full_pipeline(sample_size=sample_size, show_gui=show_gui)
    elif choice == "3":
        try:
            s_input = input("Enter sample size [default 30,000]: ").strip()
            sample_size = int(s_input.replace(",", "")) if s_input else 30_000
        except (ValueError, EOFError):
            sample_size = 30_000
        run_full_pipeline(sample_size=sample_size, show_gui=show_gui)
    else:
        run_live_flow_triage(show_gui=show_gui)


if __name__ == "__main__":
    main()
