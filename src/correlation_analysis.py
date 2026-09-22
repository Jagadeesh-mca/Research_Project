"""
correlation_analysis.py
--------------------------
Redundancy analysis and correlation-based feature selection: computes
pairwise feature correlation and drops one feature from any pair
correlated above a threshold (default 0.90), on the TRAINING split only
(to avoid leaking test-set structure into the feature-selection decision).

This is a static, model-agnostic reduction step that runs BEFORE
SHAP-driven reduction (feature_reduction.py). Running both means the
pipeline eliminates redundant/duplicated signal first (correlation), then
additionally ranks and trims by actual predictive contribution (SHAP) --
two complementary reduction strategies rather than one.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def compute_correlation_matrix(X: pd.DataFrame) -> pd.DataFrame:
    return X.corr()


def drop_correlated_features(X_train, X_test, feature_cols, threshold=0.90):
    """
    For each pair of features correlated above `threshold`, drops the
    second feature in the pair (keeps the first, alphabetically/by column
    order) -- a standard greedy redundancy-removal rule.
    Returns the reduced train/test frames, the kept feature list, the
    dropped feature list, and the full correlation matrix (for plotting).
    """
    corr = X_train[feature_cols].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if (upper[col] > threshold).any()]
    kept = [c for c in feature_cols if c not in to_drop]
    return X_train[kept], X_test[kept], kept, to_drop, corr


def plot_correlation_heatmap(corr: pd.DataFrame, out_path: str, dropped=None, show_plot=True):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0, square=True,
                xticklabels=True, yticklabels=True, cbar_kws={"shrink": 0.7})
    title = "Feature Correlation Matrix"
    if dropped:
        title += f" ({len(dropped)} Multicollinear Features Dropped at |r| > 0.90)"
    plt.title(title, fontsize=13, fontweight="bold", pad=12)
    plt.xticks(fontsize=6, rotation=90)
    plt.yticks(fontsize=6)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    if show_plot:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    import joblib
    split = joblib.load(OUTPUTS_DIR / "split.joblib")

    X_train_r, X_test_r, kept, dropped, corr = drop_correlated_features(
        split["X_train"], split["X_test"], split["feature_cols"], threshold=0.90
    )
    print(f"Features before: {len(split['feature_cols'])}, after: {len(kept)}")
    print(f"Dropped ({len(dropped)}): {dropped}")

    plot_correlation_heatmap(corr, OUTPUTS_DIR / "correlation_heatmap.png", dropped)
