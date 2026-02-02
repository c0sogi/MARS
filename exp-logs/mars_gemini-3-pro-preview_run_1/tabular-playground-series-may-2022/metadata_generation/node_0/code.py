import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Configuration
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
RANDOM_STATE = 42
VAL_SIZE = 0.2


def main():
    print("Starting metadata generation...")

    # 1. Setup Metadata Directory
    if not os.path.exists(METADATA_DIR):
        os.makedirs(METADATA_DIR)
        print(f"Created directory: {METADATA_DIR}")

    # 2. Load Raw Data
    train_path = os.path.join(INPUT_DIR, "train.csv")
    test_path = os.path.join(INPUT_DIR, "test.csv")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"{train_path} not found.")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"{test_path} not found.")

    print("Loading datasets...")
    df_train_full = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    # Add source_path to satisfy metadata requirements regarding file paths
    # These paths are relative to the ./input directory
    df_train_full["source_path"] = "train.csv"
    df_test["source_path"] = "test.csv"

    # 3. Create Validation Split
    # Using Stratified Shuffle Split as per requirements
    print(f"Splitting training data (Stratified, {1-VAL_SIZE}:{VAL_SIZE})...")

    train_df, val_df = train_test_split(
        df_train_full,
        test_size=VAL_SIZE,
        stratify=df_train_full["target"],
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    # 4. Save Metadata Files
    # Saving the actual data splits as CSVs allows efficient loading by downstream scripts
    print("Saving metadata files...")
    train_meta_path = os.path.join(METADATA_DIR, "train.csv")
    val_meta_path = os.path.join(METADATA_DIR, "val.csv")
    test_meta_path = os.path.join(METADATA_DIR, "test.csv")

    train_df.to_csv(train_meta_path, index=False)
    val_df.to_csv(val_meta_path, index=False)
    df_test.to_csv(test_meta_path, index=False)

    # 5. Verification and Checks
    print("\nPerforming verification checks...")

    # Load back the data
    meta_train = pd.read_csv(train_meta_path)
    meta_val = pd.read_csv(val_meta_path)
    meta_test = pd.read_csv(test_meta_path)

    # Summary Statistics
    print("-" * 30)
    print("Dataset Summary:")
    print(f"Train Rows: {len(meta_train)}, Cols: {meta_train.shape[1]}")
    print(f"Val Rows:   {len(meta_val)}, Cols: {meta_val.shape[1]}")
    print(f"Test Rows:  {len(meta_test)}, Cols: {meta_test.shape[1]}")

    train_target_mean = meta_train["target"].mean()
    val_target_mean = meta_val["target"].mean()

    print(f"Train Target Mean: {train_target_mean:.5f}")
    print(f"Val Target Mean:   {val_target_mean:.5f}")
    print("-" * 30)

    # Check 1: File Path Resolution
    # Check 1000 random relative file paths from the metadata
    print("Checking file path resolution...")
    all_paths = pd.concat(
        [meta_train["source_path"], meta_val["source_path"], meta_test["source_path"]]
    )

    sample_paths = all_paths.sample(
        n=min(1000, len(all_paths)), random_state=RANDOM_STATE
    )

    missing_count = 0
    missing_samples = []

    for relative_path in sample_paths:
        full_path = os.path.join(INPUT_DIR, relative_path)
        if not os.path.exists(full_path):
            missing_count += 1
            if len(missing_samples) < 5:
                missing_samples.append(relative_path)

    missing_ratio = missing_count / len(sample_paths)
    print(f"Missing file ratio: {missing_ratio:.4f}")

    if missing_ratio > 0.5:
        print("Sample of missing paths:", missing_samples)
        raise FileNotFoundError(
            f"Missing file ratio {missing_ratio} exceeds limit of 0.5"
        )

    # Check 2: Validation Split Requirements
    print("Verifying split logic...")

    # Verify Ratio
    total_train_samples = len(meta_train) + len(meta_val)
    observed_val_ratio = len(meta_val) / total_train_samples

    # Allow small floating point tolerance
    if not (0.199 <= observed_val_ratio <= 0.201):
        raise AssertionError(
            f"Validation split ratio {observed_val_ratio:.4f} is not 0.20"
        )

    # Verify Stratification
    # The difference in means should be very small
    diff = abs(train_target_mean - val_target_mean)
    if diff > 0.01:
        raise AssertionError(
            f"Stratification failed. Target mean difference {diff:.5f} is too large."
        )

    print("\nSuccess! Metadata generation and verification complete.")


if __name__ == "__main__":
    main()
