import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Configuration
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
TRAIN_FILE = "train.json"
TEST_FILE = "test.json"
RANDOM_STATE = 42
VAL_SIZE = 0.2


def load_jsonl(file_path):
    """Reads a JSONL file into a Pandas DataFrame."""
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)


def check_file_paths(df, col_name, base_dir):
    """Checks if a random sample of file paths in a column exist."""
    paths = df[col_name].dropna().tolist()
    if not paths:
        return

    # Check up to 1000 paths
    n_samples = min(1000, len(paths))
    sampled_paths = np.random.choice(paths, n_samples, replace=False)

    missing_count = 0
    sample_missing = []

    for path in sampled_paths:
        full_path = os.path.join(base_dir, path)
        if not os.path.exists(full_path):
            missing_count += 1
            if len(sample_missing) < 5:
                sample_missing.append(path)

    ratio = missing_count / n_samples
    if ratio > 0.5:
        print(f"Sample missing paths: {sample_missing}")
        raise FileNotFoundError(
            f"Missing file ratio {ratio:.2f} exceeds threshold 0.5 for column {col_name}"
        )


def main():
    # 1. Setup directories
    os.makedirs(METADATA_DIR, exist_ok=True)

    # 2. Load Data
    print("Loading data...")
    train_full_path = os.path.join(INPUT_DIR, TRAIN_FILE)
    test_path = os.path.join(INPUT_DIR, TEST_FILE)

    df_train_full = load_jsonl(train_full_path)
    df_test = load_jsonl(test_path)

    # Add source_file column for path verification
    df_train_full["source_file"] = TRAIN_FILE
    df_test["source_file"] = TEST_FILE

    # 3. Create Validation Split
    # Stratify by SN_filter if available (indicates signal quality), otherwise random
    stratify_col = None
    if "SN_filter" in df_train_full.columns:
        if df_train_full["SN_filter"].nunique() > 1:
            stratify_col = df_train_full["SN_filter"]
            print("Stratifying split based on SN_filter.")

    print(f"Splitting training data (Val size: {VAL_SIZE})...")
    df_train, df_val = train_test_split(
        df_train_full,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=stratify_col,
    )

    # 4. Save Metadata
    # Using Parquet to handle list columns (reactivity, deg_*, etc.) efficiently
    print("Saving metadata to Parquet...")
    df_train.to_parquet(os.path.join(METADATA_DIR, "train.parquet"), index=False)
    df_val.to_parquet(os.path.join(METADATA_DIR, "val.parquet"), index=False)
    df_test.to_parquet(os.path.join(METADATA_DIR, "test.parquet"), index=False)

    # 5. Verification
    print("\nPerforming verification checks...")

    # Reload data
    df_train_check = pd.read_parquet(os.path.join(METADATA_DIR, "train.parquet"))
    df_val_check = pd.read_parquet(os.path.join(METADATA_DIR, "val.parquet"))
    df_test_check = pd.read_parquet(os.path.join(METADATA_DIR, "test.parquet"))

    # Print Summary Statistics
    print("-" * 30)
    print(f"Train samples: {len(df_train_check)}")
    print(f"Val samples:   {len(df_val_check)}")
    print(f"Test samples:  {len(df_test_check)}")
    print("-" * 30)
    print(f"Train columns: {list(df_train_check.columns[:5])} ...")

    # Verify Split Ratio
    total_train = len(df_train_check) + len(df_val_check)
    val_ratio = len(df_val_check) / total_train
    print(f"Actual Validation Ratio: {val_ratio:.4f}")

    if not (0.19 <= val_ratio <= 0.21):
        raise AssertionError(
            f"Validation ratio {val_ratio:.4f} is outside acceptable range (0.2 +/- 0.01)"
        )

    # Verify No Leakage
    train_ids = set(df_train_check["id"])
    val_ids = set(df_val_check["id"])
    overlap = train_ids.intersection(val_ids)
    if overlap:
        raise AssertionError(
            f"Data leakage detected! {len(overlap)} IDs found in both train and val."
        )

    # Verify File Paths
    print("Checking file paths...")
    check_file_paths(df_train_check, "source_file", INPUT_DIR)
    check_file_paths(df_val_check, "source_file", INPUT_DIR)
    check_file_paths(df_test_check, "source_file", INPUT_DIR)

    print("\nAll checks passed successfully.")


if __name__ == "__main__":
    main()
