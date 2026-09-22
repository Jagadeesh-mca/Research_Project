"""
generate_demo_data.py
----------------------
Produces a synthetic network-flow dataset that mirrors the SCHEMA and
statistical structure of CSE-CIC-IDS2018 (same column names, same rough
feature ranges/correlations, same benign/malicious class imbalance and
attack-category mix).

WHY THIS EXISTS
The real CSE-CIC-IDS2018 dataset is ~6-16 GB, hosted on AWS S3 by the
Canadian Institute for Cybersecurity, and is not reachable from this
sandboxed build environment. Every other module in this project
(preprocessing, training, SHAP, stability, dashboard) is written against
this exact column schema, so swapping this synthetic generator for the
real CSVs is a one-line change -- see load_real_dataset() at the bottom.

Any numbers produced downstream from THIS generator are a proof-of-concept
demonstration that the pipeline works end-to-end. They are not a claim
about real-world intrusion-detection performance.
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(42)

# Core flow-level features actually present in CSE-CIC-IDS2018
# (CICFlowMeter output columns), trimmed to the set most relevant here.
FEATURE_COLUMNS = [
    "Flow_Duration",
    "Total_Fwd_Packets",
    "Total_Bwd_Packets",
    "Total_Length_Fwd_Packets",
    "Total_Length_Bwd_Packets",
    "Fwd_Packet_Length_Mean",
    "Bwd_Packet_Length_Mean",
    "Flow_Bytes_s",
    "Flow_Packets_s",
    "Flow_IAT_Mean",
    "Flow_IAT_Std",
    "Fwd_IAT_Mean",
    "Bwd_IAT_Mean",
    "Fwd_PSH_Flags",
    "SYN_Flag_Count",
    "ACK_Flag_Count",
    "RST_Flag_Count",
    "Destination_Port",
    "Packet_Length_Variance",
    "Average_Packet_Size",
    "Subflow_Fwd_Bytes",
    "Active_Mean",
    "Idle_Mean",
]

ATTACK_CATEGORIES = ["DDoS", "DoS", "Brute_Force", "Bot", "Infiltration", "Web_Attack"]


def _benign_block(n):
    """Benign traffic: moderate, fairly uniform flow characteristics."""
    return {
        "Flow_Duration": RNG.gamma(2.0, 50_000, n),
        "Total_Fwd_Packets": RNG.poisson(8, n) + 1,
        "Total_Bwd_Packets": RNG.poisson(7, n) + 1,
        "Total_Length_Fwd_Packets": RNG.gamma(2.0, 300, n),
        "Total_Length_Bwd_Packets": RNG.gamma(2.0, 280, n),
        "Fwd_Packet_Length_Mean": RNG.normal(350, 80, n).clip(0),
        "Bwd_Packet_Length_Mean": RNG.normal(340, 80, n).clip(0),
        "Flow_Bytes_s": RNG.gamma(2.0, 4_000, n),
        "Flow_Packets_s": RNG.gamma(2.0, 15, n),
        "Flow_IAT_Mean": RNG.gamma(2.0, 20_000, n),
        "Flow_IAT_Std": RNG.gamma(2.0, 8_000, n),
        "Fwd_IAT_Mean": RNG.gamma(2.0, 18_000, n),
        "Bwd_IAT_Mean": RNG.gamma(2.0, 17_000, n),
        "Fwd_PSH_Flags": RNG.binomial(1, 0.2, n),
        "SYN_Flag_Count": RNG.binomial(1, 0.3, n),
        "ACK_Flag_Count": RNG.poisson(3, n),
        "RST_Flag_Count": RNG.binomial(1, 0.05, n),
        "Destination_Port": RNG.choice([80, 443, 22, 53, 8080, 3389], n),
        "Packet_Length_Variance": RNG.gamma(2.0, 500, n),
        "Average_Packet_Size": RNG.normal(300, 60, n).clip(0),
        "Subflow_Fwd_Bytes": RNG.gamma(2.0, 300, n),
        "Active_Mean": RNG.gamma(2.0, 10_000, n),
        "Idle_Mean": RNG.gamma(2.0, 30_000, n),
    }


def _malicious_block(n, category):
    """
    Malicious traffic: shifted distributions per category so that a
    classifier -- and SHAP -- have real, category-specific signal to find.
    """
    base = _benign_block(n)
    if category == "DDoS":
        base["Flow_Packets_s"] = RNG.gamma(2.0, 400, n)         # flood
        base["Flow_Bytes_s"] = RNG.gamma(2.0, 60_000, n)
        base["Flow_Duration"] = RNG.gamma(1.2, 2_000, n)         # short bursts
        base["SYN_Flag_Count"] = RNG.binomial(1, 0.9, n)
        base["Total_Fwd_Packets"] = RNG.poisson(150, n)
    elif category == "DoS":
        base["Flow_Packets_s"] = RNG.gamma(2.0, 200, n)
        base["Flow_Duration"] = RNG.gamma(1.5, 5_000, n)
        base["SYN_Flag_Count"] = RNG.binomial(1, 0.8, n)
    elif category == "Brute_Force":
        base["Destination_Port"] = RNG.choice([22, 21, 3389, 3306], n)
        base["Total_Fwd_Packets"] = RNG.poisson(30, n)
        base["Flow_IAT_Mean"] = RNG.gamma(2.0, 500, n)           # rapid repeats
        base["RST_Flag_Count"] = RNG.binomial(1, 0.5, n)
    elif category == "Bot":
        base["Idle_Mean"] = RNG.gamma(2.0, 150_000, n)           # beaconing
        base["Active_Mean"] = RNG.gamma(1.5, 2_000, n)
        base["Flow_Duration"] = RNG.gamma(1.0, 300_000, n)
    elif category == "Infiltration":
        base["Total_Length_Bwd_Packets"] = RNG.gamma(3.0, 2_000, n)  # exfil
        base["Bwd_Packet_Length_Mean"] = RNG.normal(1200, 200, n).clip(0)
    elif category == "Web_Attack":
        base["Destination_Port"] = np.full(n, 80)
        base["Fwd_PSH_Flags"] = RNG.binomial(1, 0.8, n)
        base["Total_Fwd_Packets"] = RNG.poisson(20, n)
    return base


def generate(n_benign=45_000, n_malicious=5_000, seed=42):
    """
    Returns a DataFrame shaped like CSE-CIC-IDS2018:
    ~90% benign / ~10% malicious (mirrors the real dataset's imbalance),
    malicious split across the attack categories CIC-IDS2018 actually
    contains.
    """
    global RNG
    RNG = np.random.default_rng(seed)

    frames = [pd.DataFrame(_benign_block(n_benign))]
    frames[0]["Label"] = "Benign"
    frames[0]["Attack_Category"] = "Benign"

    per_cat = n_malicious // len(ATTACK_CATEGORIES)
    for cat in ATTACK_CATEGORIES:
        block = pd.DataFrame(_malicious_block(per_cat, cat))
        block["Label"] = "Malicious"
        block["Attack_Category"] = cat
        frames.append(block)

    df = pd.concat(frames, ignore_index=True)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)  # shuffle

    # A few injected near-duplicate benign flows that resemble low-rate
    # brute-force / scanning traffic -- these are what should surface later
    # in the false-positive analysis.
    n_hard = int(0.01 * n_benign)
    hard = pd.DataFrame(_benign_block(n_hard))
    hard["Total_Fwd_Packets"] = RNG.poisson(25, n_hard)
    hard["SYN_Flag_Count"] = RNG.binomial(1, 0.6, n_hard)
    hard["Label"] = "Benign"
    hard["Attack_Category"] = "Benign"
    df = pd.concat([df, hard], ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    return df


def load_real_dataset(csv_paths, drop_leakage_columns=True):
    """
    Drop-in replacement for generate(). Point this at the official
    CSE-CIC-IDS2018 CSVs (per-day traffic files from
    https://www.unb.ca/cic/datasets/ids-2018.html) once downloaded outside
    this environment. Handles the real-world quirks these files actually
    have, so every downstream module (preprocessing, training, SHAP,
    false-positive analysis, feature reduction, stability) works unchanged:

    - Real files use "Dst Port", "Flow Duration", etc. -> normalised to
      Dst_Port, Flow_Duration (spaces to underscores) to match this
      project's convention.
    - The real "Label" column is multi-class (e.g. "Benign",
      "FTP-BruteForce", "SSH-Bruteforce"), not the binary Benign/Malicious
      the rest of the pipeline expects. This function preserves the raw
      value as Attack_Category and derives a binary Label from it.
    - Some CIC-IDS2018 CSVs contain duplicated header rows in the middle
      of the file (an artefact of how multiple days were concatenated by
      the dataset authors) -- rows where a value equals its own column
      name are detected and dropped, and every feature column is coerced
      to numeric (non-numeric junk becomes NaN and is imputed in clean()).
    - Source/Destination IP, Flow ID, and Timestamp are dropped by default
      (drop_leakage_columns=True): keeping raw host identifiers lets a
      model memorise "this IP is always the attacker" instead of learning
      generalisable traffic behaviour, which inflates offline accuracy but
      collapses on unseen hosts. Set to False only if you specifically
      want to inspect these columns yourself.
    """
    dfs = [pd.read_csv(p, low_memory=False) for p in csv_paths]
    df = pd.concat(dfs, ignore_index=True)
    df.columns = [c.strip().replace(" ", "_").replace("/", "_") for c in df.columns]

    if "Label" not in df.columns:
        raise ValueError(
            f"No 'Label' column found after normalising headers. "
            f"Columns seen: {list(df.columns)[:10]}..."
        )

    # Drop embedded duplicate-header rows (value equals its own column name)
    df = df[df["Label"] != "Label"].copy()

    leakage_cols = ["Flow_ID", "Src_IP", "Source_IP", "Dst_IP", "Destination_IP",
                     "Timestamp", "Src_Port"]
    if drop_leakage_columns:
        df = df.drop(columns=[c for c in leakage_cols if c in df.columns])

    # Preserve the original multi-class label, derive the binary target
    # the rest of the pipeline expects.
    df["Attack_Category"] = df["Label"].astype(str).str.strip()
    df["Label"] = np.where(df["Attack_Category"].str.lower() == "benign",
                            "Benign", "Malicious")

    # Coerce every remaining feature column to numeric; junk -> NaN,
    # cleaned up by preprocessing.clean()'s median imputation.
    feature_cols = [c for c in df.columns if c not in ("Label", "Attack_Category")]
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["Label"]).reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = generate()
    out_path = DATA_DIR / "demo_traffic.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df):,} rows -> {out_path}")
    print(df["Label"].value_counts())
    print(df["Attack_Category"].value_counts())
