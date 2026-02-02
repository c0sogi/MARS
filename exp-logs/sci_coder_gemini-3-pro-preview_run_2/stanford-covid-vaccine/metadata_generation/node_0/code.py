import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Constants
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
RANDOM_STATE = 42
VAL_SIZE = 0.2


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def check_file_paths(df, sample_size=1000):
    """
    Checks if file paths in the dataframe exist.
    Assumes a 'filepath' column exists relative to INPUT_DIR.
    """
    if "filepath" not in df.columns:
        return

    # Sample paths
    n = min(len(df), sample_size)
    sample_paths = df["filepath"].sample(n=n, random_state=RANDOM_STATE).values

    missing_count = 0
    missing_samples = []

    for rel_path in sample_paths:
        full_path = os.path.join(INPUT_DIR, rel_path)
        if not os.path.exists(full_path):
            missing_count += 1
            if len(missing_samples) < 5:
                missing_samples.append(rel_path)

    missing_ratio = missing_count / n

    if missing_ratio > 0.5:
        print(f"Sample of missing paths: {missing_samples}")
        raise FileNotFoundError(
            f"Missing file ratio {missing_ratio:.2f} exceeds threshold of 0.5"
        )

    print(f"File path check passed. Missing ratio: {missing_ratio:.2f}")


def verify_stratification(train_df, val_df, stratify_col):
    """
    Verifies that the distribution of the stratification column is similar in train and val.
    """
    train_dist = train_df[stratify_col].value_counts(normalize=True).sort_index()
    val_dist = val_df[stratify_col].value_counts(normalize=True).sort_index()

    print("\nStratification Distribution (Train vs Val):")
    comparison = pd.DataFrame({"Train": train_dist, "Val": val_dist})
    print(comparison)

    # Check if the maximum difference in proportions for any bin is acceptable (e.g., < 0.05)
    diff = (train_dist - val_dist).abs().max()
    print(f"Max difference in stratification bin proportions: {diff:.4f}")

    if diff > 0.05:
        raise AssertionError(f"Stratification failed. Max difference {diff:.4f} > 0.05")
    print("Stratification verification passed.")


def main():
    ensure_dir(METADATA_DIR)

    # 1. Load Data
    print("Loading data...")
    train_path = os.path.join(INPUT_DIR, "train.json")
    test_path = os.path.join(INPUT_DIR, "test.json")

    # Read JSON lines
    df_train_full = pd.read_json(train_path, lines=True)
    df_test = pd.read_json(test_path, lines=True)

    # Add filepath metadata
    df_train_full["filepath"] = "train.json"
    df_test["filepath"] = "test.json"

    # 2. Preprocessing & Stratification
    # We stratify based on the mean reactivity to ensure balanced target distribution
    # Calculate mean reactivity for each sample
    # Note: reactivity is a list of floats
    print("Calculating stratification bins...")

    # Helper to get mean safely
    def get_mean_reactivity(x):
        if isinstance(x, list) and len(x) > 0:
            return np.mean(x)
        return 0.0

    df_train_full["mean_reactivity"] = df_train_full["reactivity"].apply(
        get_mean_reactivity
    )

    # Binning into quantiles
    num_bins = 10
    # Use qcut to get equal-sized bins, handle duplicates if many zeros
    try:
        df_train_full["stratify_bin"] = pd.qcut(
            df_train_full["mean_reactivity"],
            q=num_bins,
            labels=False,
            duplicates="drop",
        )
    except ValueError:
        # Fallback if data is too skewed for qcut
        df_train_full["stratify_bin"] = pd.cut(
            df_train_full["mean_reactivity"], bins=num_bins, labels=False
        )

    # 3. Split Data
    print("Splitting data...")
    train_df, val_df = train_test_split(
        df_train_full,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=df_train_full["stratify_bin"],
    )

    # 4. Save Metadata
    print("Saving metadata...")
    train_csv_path = os.path.join(METADATA_DIR, "train.csv")
    val_csv_path = os.path.join(METADATA_DIR, "val.csv")
    test_csv_path = os.path.join(METADATA_DIR, "test.csv")

    train_df.to_csv(train_csv_path, index=False)
    val_df.to_csv(val_csv_path, index=False)
    df_test.to_csv(test_csv_path, index=False)

    print(f"Saved train.csv ({len(train_df)} rows)")
    print(f"Saved val.csv ({len(val_df)} rows)")
    print(f"Saved test.csv ({len(df_test)} rows)")

    # 5. Validation and Checks
    print("\nPerforming validation checks...")

    # Reload to verify file integrity
    train_df_loaded = pd.read_csv(train_csv_path)
    val_df_loaded = pd.read_csv(val_csv_path)
    test_df_loaded = pd.read_csv(test_csv_path)

    # Summary Statistics
    print("\n--- Summary Statistics ---")
    print(f"Train set shape: {train_df_loaded.shape}")
    print(f"Val set shape: {val_df_loaded.shape}")
    print(f"Test set shape: {test_df_loaded.shape}")

    print("\nTrain columns:", train_df_loaded.columns.tolist())

    # Verify Split Ratio
    total_train_val = len(train_df_loaded) + len(val_df_loaded)
    val_ratio = len(val_df_loaded) / total_train_val
    print(f"\nActual Validation Ratio: {val_ratio:.4f} (Target: {VAL_SIZE})")

    if not (0.19 < val_ratio < 0.21):
        raise AssertionError(
            f"Validation split ratio {val_ratio:.4f} is too far from {VAL_SIZE}"
        )

    # Verify Stratification
    # Note: When reloading from CSV, lists/bins might need parsing, but 'stratify_bin' is integer/float
    verify_stratification(train_df_loaded, val_df_loaded, "stratify_bin")

    # Check File Paths
    print("\nChecking file paths for Train set...")
    check_file_paths(train_df_loaded)
    print("Checking file paths for Val set...")
    check_file_paths(val_df_loaded)
    print("Checking file paths for Test set...")
    check_file_paths(test_df_loaded)

    print("\nAll checks passed successfully.")


if __name__ == "__main__":
    main()
