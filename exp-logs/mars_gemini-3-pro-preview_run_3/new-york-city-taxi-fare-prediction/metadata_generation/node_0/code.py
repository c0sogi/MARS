import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Constants
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
TRAIN_FILE = "labels.csv"
TEST_FILE = "test.csv"
RANDOM_STATE = 42
VAL_SIZE = 0.2


def main():
    # 1. Setup
    if not os.path.exists(METADATA_DIR):
        os.makedirs(METADATA_DIR)

    # 2. Load Data
    train_path = os.path.join(INPUT_DIR, TRAIN_FILE)
    print(f"Loading training data from {train_path}...")

    # Use pyarrow engine for faster reading of large CSVs
    try:
        df = pd.read_csv(train_path, engine="pyarrow")
    except Exception as e:
        print(f"Pyarrow read failed ({e}), falling back to default parser.")
        df = pd.read_csv(train_path)

    print(f"Loaded {len(df)} rows.")

    # 3. Stratification
    # For regression, we stratify by binning the continuous target variable 'fare_amount'.
    print("Preparing stratification bins...")
    # We use qcut to create quantile-based bins.
    # duplicates='drop' handles cases where multiple quantiles have the same value (e.g. minimum fare).
    try:
        stratify_bins = pd.qcut(
            df["fare_amount"], q=20, labels=False, duplicates="drop"
        )
    except Exception as e:
        print(f"Stratification binning warning: {e}. Fallback to simple cut.")
        stratify_bins = pd.cut(df["fare_amount"], bins=10, labels=False)

    # 4. Split
    print("Splitting data into training and validation sets...")
    train_df, val_df = train_test_split(
        df,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=stratify_bins,
    )

    # Free up memory from the original dataframe
    del df, stratify_bins

    # 5. Save Metadata (the split datasets)
    print("Saving metadata files...")
    train_meta_path = os.path.join(METADATA_DIR, "train.parquet")
    val_meta_path = os.path.join(METADATA_DIR, "val.parquet")
    test_meta_path = os.path.join(METADATA_DIR, "test.parquet")

    train_df.to_parquet(train_meta_path, index=False)
    val_df.to_parquet(val_meta_path, index=False)

    # Process Test Data
    test_path = os.path.join(INPUT_DIR, TEST_FILE)
    print(f"Loading and saving test data from {test_path}...")
    test_df = pd.read_csv(test_path)
    test_df.to_parquet(test_meta_path, index=False)

    # 6. Verification
    print("Performing validation checks...")

    # Reload to ensure integrity
    train_check = pd.read_parquet(train_meta_path)
    val_check = pd.read_parquet(val_meta_path)
    test_check = pd.read_parquet(test_meta_path)

    # A. Summary Statistics
    print("\n=== Dataset Summary ===")
    print(f"Training Set: {len(train_check)} samples")
    print(f"Validation Set: {len(val_check)} samples")
    print(f"Test Set: {len(test_check)} samples")

    print("\nTarget Distribution (Fare Amount):")
    print("Train:")
    print(train_check["fare_amount"].describe())
    print("Validation:")
    print(val_check["fare_amount"].describe())

    # B. Check Split Ratio
    total_samples = len(train_check) + len(val_check)
    actual_val_ratio = len(val_check) / total_samples
    print(f"\nActual Validation Ratio: {actual_val_ratio:.6f}")

    # Assert ratio is within small tolerance (e.g., due to rounding)
    if not (0.199 < actual_val_ratio < 0.201):
        raise AssertionError(
            f"Validation split ratio {actual_val_ratio} deviates from required 0.2"
        )

    # C. Check Stratification
    # We check if the mean fare amount is similar between train and val
    train_mean = train_check["fare_amount"].mean()
    val_mean = val_check["fare_amount"].mean()

    # Calculate relative difference
    rel_diff = abs(train_mean - val_mean) / (abs(train_mean) + 1e-9)
    print(f"Relative difference in target mean: {rel_diff:.6f}")

    # Assert distribution is preserved (tolerance 1%)
    if rel_diff > 0.01:
        raise AssertionError(
            "Stratification failed: Target distribution means differ significantly."
        )

    # D. File Path Check
    # The dataset is tabular and self-contained in the parquet files.
    # There are no external file paths (like image paths) to verify.
    print("No external file path columns to verify.")

    print("\nMetadata generation and verification completed successfully.")


if __name__ == "__main__":
    main()
