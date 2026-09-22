"""
preprocessing.py
-----------------
Data cleaning and preprocessing module for Cloud Intrusion Detection.
Handles infinity replacement, median imputation for skewed distributions,
zero-variance column removal, and stratified train/test splitting.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

DATA_DIR = PROJECT_ROOT / "data"


def clean(df: pd.DataFrame, drop_duplicates: bool = True, drop_zero_variance: bool = True) -> tuple[pd.DataFrame, list[str]]:
    """
    Rigorous preprocessing operations adhering to research standard and the reference paper:
    1. Drop exact duplicate rows (default True to prevent train/test contamination and inflated metrics).
    2. Replace +/- infinity values with NaN.
    3. Drop columns with zero variance (constant across all observations).
    4. Impute missing values with median (robust against severe network traffic skew).
    """
    df = df.copy()

    if drop_duplicates:
        initial_len = len(df)
        df = df.drop_duplicates()
        dropped_dups = initial_len - len(df)
        print(f"Dropped {dropped_dups:,} exact duplicate rows ({dropped_dups/initial_len*100:.2f}% of input).")

    feature_cols = [c for c in df.columns if c not in ("Label", "Attack_Category")]

    # Replace inf and -inf with NaN
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)

    # Drop rows that are completely NaN in feature columns
    df = df.dropna(subset=feature_cols, how="all")

    # Detect and drop zero-variance (constant) features
    dropped_const_cols = []
    if drop_zero_variance:
        stds = df[feature_cols].std()
        dropped_const_cols = stds[stds == 0].index.tolist()
        if dropped_const_cols:
            print(f"Dropping {len(dropped_const_cols)} zero-variance columns: {dropped_const_cols}")
            df = df.drop(columns=dropped_const_cols)
            feature_cols = [c for c in feature_cols if c not in dropped_const_cols]

    # Impute remaining NaNs with column median
    for col in feature_cols:
        if df[col].isna().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val if not pd.isna(median_val) else 0.0)

    return df.reset_index(drop=True), dropped_const_cols


def verify_data_leakage(X_train: pd.DataFrame, X_test: pd.DataFrame, 
                        y_train: np.ndarray, y_test: np.ndarray, 
                        feature_cols: list[str]) -> dict:
    """
    Performs rigorous data leakage checks:
    1. Train/Test duplicate overlap check (records appearing in both train and test)
    2. Zero-variance feature check in training split
    3. Identifier column leakage check (IP, MAC, Timestamp, Flow_ID)
    4. Target variable contamination in feature matrix
    """
    checks = {}
    
    # Check 1: Overlap between X_train and X_test
    # Hash or tuple-based comparison of train and test rows
    train_set = set(map(tuple, X_train.values))
    test_rows = list(map(tuple, X_test.values))
    overlap_count = sum(1 for row in test_rows if row in train_set)
    checks["train_test_overlap_count"] = overlap_count
    checks["train_test_overlap_pct"] = float(overlap_count / len(X_test) * 100)

    # Check 2: Identifier leakage
    forbidden_terms = ["ip", "timestamp", "flow_id", "mac", "src_port", "dst_ip", "src_ip"]
    leaked_identifiers = [c for c in feature_cols if any(t in c.lower() for t in forbidden_terms)]
    checks["leaked_identifier_columns"] = leaked_identifiers

    # Check 3: Target leakage (label in features)
    target_terms = ["label", "attack", "malicious", "class"]
    leaked_target = [c for c in feature_cols if any(t == c.lower() for t in target_terms)]
    checks["target_leakage_columns"] = leaked_target

    # Check 4: Zero variance in train
    stds = X_train.std()
    zero_var_train = stds[stds == 0].index.tolist()
    checks["zero_variance_train_columns"] = zero_var_train

    is_clean = (overlap_count == 0 and len(leaked_identifiers) == 0 and 
                len(leaked_target) == 0 and len(zero_var_train) == 0)
    checks["leakage_free"] = is_clean

    return checks


def encode_and_split(df: pd.DataFrame, test_size: float = 0.20, seed: int = 42, 
                     apply_minmax: bool = True) -> dict:
    """
    Encodes the binary label (Benign=0, Malicious=1), splits features and targets,
    and returns a stratified train/test split.
    Optionally applies Min-Max normalization fitted strictly on X_train.
    """
    from sklearn.preprocessing import MinMaxScaler
    feature_cols = [c for c in df.columns if c not in ("Label", "Attack_Category")]

    le = LabelEncoder()
    y = le.fit_transform(df["Label"])  # Benign=0, Malicious=1

    X = df[feature_cols].copy()
    categories = df["Attack_Category"] if "Attack_Category" in df.columns else df["Label"]

    X_train, X_test, y_train, y_test, cat_train, cat_test = train_test_split(
        X, y, categories,
        test_size=test_size, random_state=seed, stratify=y,
    )

    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    cat_train = cat_train.reset_index(drop=True)
    cat_test = cat_test.reset_index(drop=True)

    scaler = None
    if apply_minmax:
        scaler = MinMaxScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_cols)
    else:
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()

    # Leakage verification
    leakage_report = verify_data_leakage(X_train, X_test, y_train, y_test, feature_cols)

    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "X_train_raw": X_train,
        "X_test_raw": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "cat_train": cat_train,
        "cat_test": cat_test,
        "feature_cols": feature_cols,
        "label_encoder": le,
        "scaler": scaler,
        "leakage_report": leakage_report,
    }
