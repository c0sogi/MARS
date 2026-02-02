import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

# Constants
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
RANDOM_STATE = 42
VAL_SIZE = 0.2


def generate_metadata():
    print("Starting metadata generation...")

    # Ensure metadata directory exists
    os.makedirs(METADATA_DIR, exist_ok=True)

    # File paths
    train_path = os.path.join(INPUT_DIR, "train.csv")
    test_path = os.path.join(INPUT_DIR, "test.csv")

    # Load raw data
    print(f"Loading data from {INPUT_DIR}...")
    df_train_full = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    # Add relative source path for compliance and tracking
    # Paths must be relative to ./input
    df_train_full["source_path"] = "train.csv"
    df_test["source_path"] = "test.csv"

    # Perform Stratified Split
    print("Performing stratified split (80/20)...")
    if "target" not in df_train_full.columns:
        raise ValueError("Target column not found in training data.")

    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=VAL_SIZE, random_state=RANDOM_STATE
    )

    # We split based on the target
    for train_idx, val_idx in splitter.split(df_train_full, df_train_full["target"]):
        df_train = df_train_full.iloc[train_idx].copy()
        df_val = df_train_full.iloc[val_idx].copy()

    # Save metadata files
    print("Saving metadata files...")
    train_meta_path = os.path.join(METADATA_DIR, "train.csv")
    val_meta_path = os.path.join(METADATA_DIR, "val.csv")
    test_meta_path = os.path.join(METADATA_DIR, "test.csv")

    df_train.to_csv(train_meta_path, index=False)
    df_val.to_csv(val_meta_path, index=False)
    df_test.to_csv(test_meta_path, index=False)

    print("Metadata generation complete.")
    return train_meta_path, val_meta_path, test_meta_path


def verify_metadata(train_path, val_path, test_path):
    print("\nStarting verification...")

    # Load generated metadata
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    # 1. Print Summary Statistics
    print("\n=== Summary Statistics ===")
    print(f"Train set shape: {df_train.shape}")
    print(f"Val set shape:   {df_val.shape}")
    print(f"Test set shape:  {df_test.shape}")

    print("\nTrain Target Distribution:")
    print(df_train["target"].value_counts(normalize=True))
    print("\nVal Target Distribution:")
    print(df_val["target"].value_counts(normalize=True))

    # 2. Check File Paths
    # We check the 'source_path' column which contains paths relative to ./input
    print("\nChecking file path resolution...")

    def check_paths(df, name):
        if "source_path" not in df.columns:
            return

        # Select up to 1000 random paths
        n_samples = min(1000, len(df))
        sample_paths = df["source_path"].sample(n=n_samples, random_state=RANDOM_STATE)

        missing_count = 0
        missing_samples = []

        for rel_path in sample_paths:
            full_path = os.path.join(INPUT_DIR, rel_path)
            if not os.path.exists(full_path):
                missing_count += 1
                if len(missing_samples) < 5:
                    missing_samples.append(rel_path)

        missing_ratio = missing_count / n_samples
        print(f"[{name}] Missing file ratio: {missing_ratio:.4f}")

        if missing_ratio > 0.5:
            print(f"Sample missing paths: {missing_samples}")
            raise FileNotFoundError(
                f"More than 50% of file paths in {name} are invalid."
            )

    check_paths(df_train, "Train")
    check_paths(df_val, "Val")
    check_paths(df_test, "Test")

    # 3. Verify Validation Split Requirements
    print("\nVerifying split requirements...")

    # Check Ratio
    total_train_val = len(df_train) + len(df_val)
    val_ratio = len(df_val) / total_train_val
    print(f"Actual Validation Ratio: {val_ratio:.4f}")

    # Allow a tiny margin of error for rounding/integer division
    if not (0.19 < val_ratio < 0.21):
        raise AssertionError(
            f"Validation split ratio {val_ratio:.4f} is not approximately 0.20"
        )

    # Check Stratification
    train_target_mean = df_train["target"].mean()
    val_target_mean = df_val["target"].mean()
    diff = abs(train_target_mean - val_target_mean)

    print(f"Train Target Mean: {train_target_mean:.5f}")
    print(f"Val Target Mean:   {val_target_mean:.5f}")
    print(f"Difference:        {diff:.5f}")

    # Assert stratification is successful (difference should be very small)
    if diff > 0.01:  # 1% tolerance
        raise AssertionError(
            "Stratified sampling failed: Target distributions differ significantly."
        )

    print("\nAll checks passed successfully.")


if __name__ == "__main__":
    try:
        t_path, v_path, te_path = generate_metadata()
        verify_metadata(t_path, v_path, te_path)
    except Exception as e:
        print(f"\nERROR: {e}")
        exit(1)
