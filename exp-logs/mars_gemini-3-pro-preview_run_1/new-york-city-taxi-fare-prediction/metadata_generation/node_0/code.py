import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import random

# Configuration
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
TRAIN_FILE = "labels.csv"  # As per file listing
TEST_FILE = "test.csv"
RANDOM_STATE = 42
VAL_SIZE = 0.2


def check_file_paths(df, sample_size=1000):
    """
    Checks if any column looks like a file path relative to ./input.
    If found, verifies a random sample of paths exists.
    """
    path_cols = []
    # Heuristic: check string columns for 'input/' or common extensions
    for col in df.select_dtypes(include=["object", "string"]).columns:
        # Check first valid value
        sample_val = df[col].dropna().iloc[0] if not df[col].dropna().empty else ""
        if isinstance(sample_val, str) and (
            sample_val.startswith("input/") or sample_val.startswith("./input/")
        ):
            path_cols.append(col)

    if not path_cols:
        return

    print(f"Checking file paths in columns: {path_cols}")
    for col in path_cols:
        paths = df[col].dropna().tolist()
        if not paths:
            continue

        # Sample paths
        n_check = min(len(paths), sample_size)
        check_paths = random.sample(paths, n_check)

        missing_count = 0
        missing_samples = []

        for p in check_paths:
            # Resolve relative to current working directory (assuming paths are relative to root)
            # or relative to input. The prompt says paths are relative to ./input.
            # Usually metadata stores 'input/img.jpg'.
            if not os.path.exists(p):
                missing_count += 1
                if len(missing_samples) < 5:
                    missing_samples.append(p)

        missing_ratio = missing_count / n_check
        print(f"Column '{col}': Missing ratio = {missing_ratio:.4f}")

        if missing_ratio > 0.5:
            print("Sample missing paths:", missing_samples)
            raise FileNotFoundError(f"More than 50% of files missing in column {col}")


def generate_metadata():
    print("Starting metadata generation...")
    os.makedirs(METADATA_DIR, exist_ok=True)

    # --- Process Test Data ---
    test_path = os.path.join(INPUT_DIR, TEST_FILE)
    if os.path.exists(test_path):
        print(f"Loading {TEST_FILE}...")
        df_test = pd.read_csv(test_path)
        test_out_path = os.path.join(METADATA_DIR, "test.parquet")
        df_test.to_parquet(test_out_path, index=False)
        print(f"Saved test metadata to {test_out_path} ({df_test.shape})")
    else:
        raise FileNotFoundError(f"{TEST_FILE} not found in {INPUT_DIR}")

    # --- Process Training Data ---
    train_path = os.path.join(INPUT_DIR, TRAIN_FILE)
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"{TRAIN_FILE} not found in {INPUT_DIR}")

    print(f"Loading {TRAIN_FILE} (this may take a moment)...")
    # Use pyarrow engine for faster reading of large CSVs
    try:
        df = pd.read_csv(train_path, engine="pyarrow")
    except Exception as e:
        print(f"PyArrow engine failed ({e}), falling back to default...")
        df = pd.read_csv(train_path)

    print(f"Loaded training data: {df.shape}")

    # --- Stratified Split ---
    print("Creating stratified split based on 'fare_amount'...")

    # Create bins for continuous target stratification
    # qcut divides into equal-sized buckets. duplicates='drop' handles dense values.
    try:
        df["stratify_bin"] = pd.qcut(
            df["fare_amount"], q=20, labels=False, duplicates="drop"
        )
    except Exception as e:
        print(f"Binning failed ({e}). Using random split without stratification.")
        df["stratify_bin"] = 0

    # Fill NaNs in bin (if any) to avoid split errors
    df["stratify_bin"] = df["stratify_bin"].fillna(-1)

    train_df, val_df = train_test_split(
        df, test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=df["stratify_bin"]
    )

    # Clean up auxiliary column
    train_df = train_df.drop(columns=["stratify_bin"])
    val_df = val_df.drop(columns=["stratify_bin"])

    # Save to Parquet
    print("Saving train/val splits to metadata...")
    train_df.to_parquet(os.path.join(METADATA_DIR, "train.parquet"), index=False)
    val_df.to_parquet(os.path.join(METADATA_DIR, "val.parquet"), index=False)
    print("Metadata generation finished.")


def verify_metadata():
    print("\n--- Verifying Metadata ---")

    # Load generated metadata
    try:
        train_df = pd.read_parquet(os.path.join(METADATA_DIR, "train.parquet"))
        val_df = pd.read_parquet(os.path.join(METADATA_DIR, "val.parquet"))
        test_df = pd.read_parquet(os.path.join(METADATA_DIR, "test.parquet"))
    except Exception as e:
        raise AssertionError(f"Failed to load generated metadata files: {e}")

    # 1. Summary Statistics
    print(f"Train set shape: {train_df.shape}")
    print(f"Val set shape:   {val_df.shape}")
    print(f"Test set shape:  {test_df.shape}")

    print("\nTrain Fare Amount Stats:")
    print(train_df["fare_amount"].describe())
    print("\nVal Fare Amount Stats:")
    print(val_df["fare_amount"].describe())

    # 2. Check File Paths
    # (If columns contained paths like 'input/image_01.jpg', this would validate them)
    check_file_paths(train_df)
    check_file_paths(val_df)
    check_file_paths(test_df)

    # 3. Verify Split Requirements
    total_samples = len(train_df) + len(val_df)
    val_ratio = len(val_df) / total_samples
    print(f"\nValidation Split Ratio: {val_ratio:.4f}")

    # Assert split ratio is within small margin of 0.2
    assert 0.199 < val_ratio < 0.201, f"Validation ratio {val_ratio} deviates from 0.2"

    # 4. Verify Stratification
    # Compare means of target variable
    train_mean = train_df["fare_amount"].mean()
    val_mean = val_df["fare_amount"].mean()

    # Calculate relative difference
    # We expect them to be very close due to stratification and large N
    rel_diff = abs(train_mean - val_mean) / (abs(train_mean) + 1e-9)
    print(f"Relative difference in mean fare_amount: {rel_diff:.6f}")

    # Assert stratification success (tolerance 1%)
    if rel_diff > 0.01:
        raise AssertionError(
            f"Stratification failed: Train mean {train_mean} vs Val mean {val_mean} differ significantly."
        )

    print("\nVerification passed successfully.")


if __name__ == "__main__":
    generate_metadata()
    verify_metadata()
