"""
stability_analysis.py
------------------------
Test explanation stability: does similar traffic receive similar SHAP
explanations? Two complementary tests:

1. Neighbour consistency: for a sample of test flows, find their nearest
   neighbour (in feature space, same predicted class) and measure the
   rank correlation between their SHAP explanation vectors. High
   correlation = stable/consistent explanations for similar inputs.

2. Bootstrap re-training stability: retrain the Random Forest on several
   bootstrap resamples of the training data and check how consistent the
   GLOBAL SHAP feature ranking is across retrains (Jaccard overlap of
   top-k features, Spearman correlation of importance ranks).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import matplotlib
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

from shap_explain import build_explainer, global_feature_importance
from train_model import train_random_forest


def neighbour_explanation_consistency(explainer, X, n_samples=150, seed=42):
    """
    Evaluates local explanation stability for nearest neighbours in standardized feature space.
    Computes both Spearman rank correlation and Cosine similarity between SHAP explanation vectors.
    """
    rng = np.random.default_rng(seed)
    X_std = (X - X.mean()) / (X.std() + 1e-9)

    nn = NearestNeighbors(n_neighbors=2).fit(X_std.values)
    sample_idx = rng.choice(len(X), size=min(n_samples, len(X)), replace=False)

    spearmans = []
    cosines = []
    for i in sample_idx:
        dist, idx = nn.kneighbors(X_std.iloc[[i]].values)
        neighbour_idx = idx[0][1]
        if neighbour_idx == i:
            continue
        pair = X.iloc[[i, neighbour_idx]]
        sv = explainer.shap_values(pair)
        if isinstance(sv, list):
            sv_pos = sv[1]
        elif np.ndim(sv) == 3:
            sv_pos = sv[:, :, 1]
        else:
            sv_pos = sv

        rho, _ = spearmanr(sv_pos[0], sv_pos[1])
        cos = cosine_similarity(sv_pos[0].reshape(1, -1), sv_pos[1].reshape(1, -1))[0, 0]

        if not np.isnan(rho):
            spearmans.append(rho)
        if not np.isnan(cos):
            cosines.append(cos)

    return np.array(spearmans), np.array(cosines)


def bootstrap_ranking_stability(X_train, y_train, feature_cols, n_bootstraps=5,
                                 sample_size=1000, seed=42, n_estimators=80):
    """
    Retrains the model on bootstrap resamples and recomputes the global
    SHAP feature ranking each time to assess global ranking stability.
    """
    rng = np.random.default_rng(seed)
    rankings = []

    for b in range(n_bootstraps):
        idx = rng.choice(len(X_train), size=len(X_train), replace=True)
        Xb, yb = X_train.iloc[idx], y_train[idx]
        clf_b, _ = train_random_forest(Xb, yb, seed=seed + b, n_estimators=n_estimators)

        explainer_b = build_explainer(clf_b)
        X_sample = X_train.sample(min(sample_size, len(X_train)), random_state=seed)
        sv = explainer_b.shap_values(X_sample)
        if isinstance(sv, list):
            sv_pos = sv[1]
        elif np.ndim(sv) == 3:
            sv_pos = sv[:, :, 1]
        else:
            sv_pos = sv
        ranking = global_feature_importance(sv_pos, feature_cols)
        rankings.append(ranking)

    return rankings


def summarise_bootstrap_stability(rankings, top_k=8):
    """Jaccard overlap of top-k and top-5 feature sets + Spearman corr of full ranks, pairwise."""
    top_sets_8 = [set(r.head(top_k).index) for r in rankings]
    top_sets_5 = [set(r.head(5).index) for r in rankings]
    jaccards_8, jaccards_5, spearmans = [], [], []
    for i in range(len(rankings)):
        for j in range(i + 1, len(rankings)):
            inter_8 = len(top_sets_8[i] & top_sets_8[j])
            union_8 = len(top_sets_8[i] | top_sets_8[j])
            jaccards_8.append(inter_8 / max(union_8, 1))

            inter_5 = len(top_sets_5[i] & top_sets_5[j])
            union_5 = len(top_sets_5[i] | top_sets_5[j])
            jaccards_5.append(inter_5 / max(union_5, 1))

            common = rankings[i].index
            rho, _ = spearmanr(rankings[i][common], rankings[j].reindex(common))
            spearmans.append(rho)
    return {
        "mean_jaccard_top8": float(np.mean(jaccards_8)) if jaccards_8 else 1.0,
        "mean_jaccard_top5": float(np.mean(jaccards_5)) if jaccards_5 else 1.0,
        "mean_spearman_full_ranking": float(np.mean(spearmans)) if spearmans else 1.0,
    }


def plot_stability(spearmans, cosines, out_path, show_plot=True):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Spearman rank correlation distribution
    axes[0].hist(spearmans, bins=15, color="#2ca02c", edgecolor="black", alpha=0.8)
    axes[0].axvline(np.median(spearmans), color="darkgreen", linestyle="--", lw=2,
                    label=f"Median = {np.median(spearmans):.3f}")
    axes[0].set_xlabel("Spearman Rank Correlation ($\\rho$)", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Pair Count", fontsize=11, fontweight="bold")
    axes[0].set_title("Local Explanation Stability: Rank Correlation", fontsize=11, fontweight="bold")
    axes[0].legend(loc="upper left")
    axes[0].grid(axis="y", linestyle=":", alpha=0.6)

    # Cosine similarity distribution
    axes[1].hist(cosines, bins=15, color="#1f77b4", edgecolor="black", alpha=0.8)
    axes[1].axvline(np.median(cosines), color="navy", linestyle="--", lw=2,
                    label=f"Median = {np.median(cosines):.3f}")
    axes[1].set_xlabel("Cosine Similarity", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Pair Count", fontsize=11, fontweight="bold")
    axes[1].set_title("Local Explanation Stability: Cosine Alignment", fontsize=11, fontweight="bold")
    axes[1].legend(loc="upper left")
    axes[1].grid(axis="y", linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    if show_plot:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    clf = joblib.load(OUTPUTS_DIR / "rf_full_model.joblib")
    split = joblib.load(OUTPUTS_DIR / "split.joblib")
    explainer = build_explainer(clf)

    print("Running neighbour-consistency test...")
    spearmans, cosines = neighbour_explanation_consistency(explainer, split["X_test"], n_samples=50)
    print(f"  n pairs evaluated: {len(spearmans)}")
    print(f"  median Spearman correlation: {np.median(spearmans):.3f}")
    print(f"  mean Spearman correlation:   {np.mean(spearmans):.3f}")
    print(f"  median Cosine similarity:    {np.median(cosines):.3f}")
    print(f"  mean Cosine similarity:      {np.mean(cosines):.3f}")
    print(f"  fraction with correlation > 0.7: {(spearmans > 0.7).mean()*100:.1f}%")
    plot_stability(spearmans, cosines, OUTPUTS_DIR / "stability_neighbour.png")

    print("\nRunning bootstrap ranking-stability test (3 retrains)...")
    rankings = bootstrap_ranking_stability(
        split["X_train"], split["y_train"], split["feature_cols"],
        n_bootstraps=3, sample_size=400, n_estimators=80,
    )
    summary = summarise_bootstrap_stability(rankings, top_k=8)
    print(f"  mean Jaccard overlap of top-8 features across retrains: {summary['mean_jaccard_top8']:.3f}")
    print(f"  mean Spearman corr of full feature ranking across retrains: {summary['mean_spearman_full_ranking']:.3f}")

    joblib.dump({"spearmans": spearmans, "cosines": cosines, "bootstrap_summary": summary},
                OUTPUTS_DIR / "stability_results.joblib")
