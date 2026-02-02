import pandas as pd
import numpy as np
import os

# Configuration
INPUT_DIR = "./input"
META_DIR = "./metadata"
RANDOM_STATE = 42
TRAIN_VAL_RATIO = 0.8


def main():
    print("Starting metadata generation...")

    # Ensure metadata directory exists
    os.makedirs(META_DIR, exist_ok=True)

    # 1. Load Raw Data
    print("Loading raw data...")
    train_path = os.path.join(INPUT_DIR, "train.csv")
    test_path = os.path.join(INPUT_DIR, "test.csv")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError("Input files not found in ./input directory")

    train_raw = pd.read_csv(train_path)
    test_raw = pd.read_csv(test_path)

    # 2. Perform Group Split (Group Sampling by breath_id)
    print("Performing group split...")
    # Get unique groups
    unique_breaths = train_raw["breath_id"].unique()

    # Shuffle groups
    rng = np.random.default_rng(RANDOM_STATE)
    rng.shuffle(unique_breaths)

    # Split groups
    n_train = int(len(unique_breaths) * TRAIN_VAL_RATIO)
    train_breaths = unique_breaths[:n_train]
    val_breaths = unique_breaths[n_train:]

    # Create DataFrames based on groups
    # Using isin is efficient for filtering
    train_df = train_raw[train_raw["breath_id"].isin(train_breaths)].copy()
    val_df = train_raw[train_raw["breath_id"].isin(val_breaths)].copy()

    # 3. Save Metadata (Split Datasets)
    print("Saving metadata files...")
    train_meta_path = os.path.join(META_DIR, "train.csv")
    val_meta_path = os.path.join(META_DIR, "validation.csv")
    test_meta_path = os.path.join(META_DIR, "test.csv")

    train_df.to_csv(train_meta_path, index=False)
    val_df.to_csv(val_meta_path, index=False)
    test_raw.to_csv(test_meta_path, index=False)

    print("Metadata generation complete.")

    # 4. Verification and Checks
    print("Running verification checks...")

    # Load back the data to verify
    v_train = pd.read_csv(train_meta_path)
    v_val = pd.read_csv(val_meta_path)
    v_test = pd.read_csv(test_meta_path)

    # Summary Statistics
    print("\n=== Summary Statistics ===")
    print(
        f"Train Set: {v_train.shape[0]} samples, {v_train['breath_id'].nunique()} unique breaths"
    )
    print(
        f"Val Set:   {v_val.shape[0]} samples, {v_val['breath_id'].nunique()} unique breaths"
    )
    print(
        f"Test Set:  {v_test.shape[0]} samples, {v_test['breath_id'].nunique()} unique breaths"
    )

    print("\nTarget Distribution (Pressure):")
    print(
        f"Train Mean: {v_train['pressure'].mean():.4f}, Std: {v_train['pressure'].std():.4f}"
    )
    print(
        f"Val Mean:   {v_val['pressure'].mean():.4f}, Std: {v_val['pressure'].std():.4f}"
    )

    # Check 1: File Paths
    # Since this is tabular data and we saved the data directly, there are no external relative file paths
    # (like images) to check. We skip the "missing file ratio" check as it's not applicable to this dataset format.

    # Check 2: Validation Split Requirements
    print("\nVerifying split logic...")

    # Assert Group Split (No leakage of breath_id)
    train_ids_set = set(v_train["breath_id"].unique())
    val_ids_set = set(v_val["breath_id"].unique())
    intersection = train_ids_set.intersection(val_ids_set)

    if len(intersection) > 0:
        raise AssertionError(
            f"Data Leakage detected! {len(intersection)} breath_ids found in both train and validation sets."
        )

    # Assert Split Ratio
    total_breaths = len(train_ids_set) + len(val_ids_set)
    actual_ratio = len(train_ids_set) / total_breaths
    print(f"Actual Train Split Ratio: {actual_ratio:.4f}")

    # Allow for minor deviation due to integer division
    if not (0.79 < actual_ratio < 0.81):
        raise AssertionError(
            f"Split ratio mismatch. Expected ~0.8, got {actual_ratio:.4f}"
        )

    # Assert Labels exist
    if "pressure" not in v_train.columns or "pressure" not in v_val.columns:
        raise AssertionError(
            "Ground truth label 'pressure' missing from training or validation metadata."
        )

    print("\nAll checks passed successfully.")


if __name__ == "__main__":
    main()
