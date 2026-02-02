import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Constants
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
RANDOM_STATE = 42
VAL_SIZE = 0.2
TARGET_COL = "Cover_Type"


def generate_metadata():
    """
    Reads raw data, creates a validation split, and saves metadata (parquet files).
    """
    print("Starting metadata generation...")
    os.makedirs(METADATA_DIR, exist_ok=True)

    # --- Process Training Data ---
    train_path = os.path.join(INPUT_DIR, "train.csv")
    print(f"Loading training data from {train_path}...")
    df_train_full = pd.read_csv(train_path)

    if TARGET_COL not in df_train_full.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in train.csv")

    # Filter out classes with fewer than 2 samples to allow stratified split
    class_counts = df_train_full[TARGET_COL].value_counts()
    rare_classes = class_counts[class_counts < 2].index
    if not rare_classes.empty:
        print(
            f"Warning: Found classes with < 2 samples: {rare_classes.tolist()}. Removing them to allow stratified split."
        )
        df_train_full = df_train_full[~df_train_full[TARGET_COL].isin(rare_classes)]

    print(
        f"Splitting training data (Split ratio: {1-VAL_SIZE}:{VAL_SIZE}, Stratified)..."
    )
    train_df, val_df = train_test_split(
        df_train_full,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=df_train_full[TARGET_COL],
    )

    train_meta_path = os.path.join(METADATA_DIR, "train.parquet")
    val_meta_path = os.path.join(METADATA_DIR, "val.parquet")

    print(f"Saving training metadata to {train_meta_path}...")
    train_df.to_parquet(train_meta_path, index=False)

    print(f"Saving validation metadata to {val_meta_path}...")
    val_df.to_parquet(val_meta_path, index=False)

    # --- Process Test Data ---
    test_path = os.path.join(INPUT_DIR, "test.csv")
    print(f"Loading test data from {test_path}...")
    df_test = pd.read_csv(test_path)

    test_meta_path = os.path.join(METADATA_DIR, "test.parquet")
    print(f"Saving test metadata to {test_meta_path}...")
    df_test.to_parquet(test_meta_path, index=False)

    return train_meta_path, val_meta_path, test_meta_path


def check_file_paths(df, base_dir):
    """
    Checks if columns contain file paths and verifies their existence.
    """
    # Heuristic: Identify columns that might contain file paths
    # We look for string columns containing '/' or common extensions
    path_cols = []
    string_cols = df.select_dtypes(include=["object", "string"]).columns

    for col in string_cols:
        # Check a non-null sample
        sample_series = df[col].dropna()
        if sample_series.empty:
            continue

        sample = str(sample_series.iloc[0])
        # Simple heuristic for file paths
        if ("/" in sample) or ("." in sample and len(sample) > 4):
            path_cols.append(col)

    if not path_cols:
        print("No file path columns detected. Skipping file existence check.")
        return

    for col in path_cols:
        print(f"Verifying file paths in column: '{col}'...")
        # Randomly sample up to 1000 paths
        sample_paths = (
            df[col].sample(n=min(1000, len(df)), random_state=RANDOM_STATE).tolist()
        )

        missing_count = 0
        missing_samples = []

        for p in sample_paths:
            # Paths are relative to ./input
            full_path = os.path.join(base_dir, str(p))
            if not os.path.exists(full_path):
                missing_count += 1
                if len(missing_samples) < 5:
                    missing_samples.append(full_path)

        missing_ratio = missing_count / len(sample_paths)
        print(f"  Missing file ratio for '{col}': {missing_ratio:.4f}")

        if missing_ratio > 0.5:
            print("  Sample of missing paths:")
            for mp in missing_samples:
                print(f"    {mp}")
            raise FileNotFoundError(
                f"Validation failed: >50% of files missing in column '{col}'."
            )


def validate_metadata(train_path, val_path, test_path):
    """
    Loads generated metadata and performs validation checks.
    """
    print("\n--- Validating Metadata ---")

    # Load datasets
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)

    # 1. Summary Statistics
    print(f"Train shape: {train_df.shape}")
    print(f"Val shape:   {val_df.shape}")
    print(f"Test shape:  {test_df.shape}")

    if TARGET_COL in train_df.columns:
        print("\nTrain Class Distribution:")
        print(train_df[TARGET_COL].value_counts(normalize=True))
        print("\nValidation Class Distribution:")
        print(val_df[TARGET_COL].value_counts(normalize=True))

    # 2. Verify Stratification
    if TARGET_COL in train_df.columns:
        print("\nVerifying stratification...")
        train_dist = train_df[TARGET_COL].value_counts(normalize=True).sort_index()
        val_dist = val_df[TARGET_COL].value_counts(normalize=True).sort_index()

        # Align indices to ensure we compare same classes
        all_classes = train_dist.index.union(val_dist.index)
        train_dist = train_dist.reindex(all_classes, fill_value=0)
        val_dist = val_dist.reindex(all_classes, fill_value=0)

        diff = (train_dist - val_dist).abs()
        max_diff = diff.max()
        print(f"Max difference in class proportions: {max_diff:.6f}")

        # Assert stratification success (tolerance 1%)
        if max_diff > 0.01:
            raise AssertionError(
                f"Stratification failed. Max class proportion difference {max_diff:.6f} > 0.01"
            )
        else:
            print("Stratification check passed.")

    # 3. Check File Paths
    # Paths in metadata are relative to ./input
    print("\nChecking file paths in Training set...")
    check_file_paths(train_df, INPUT_DIR)

    print("\nChecking file paths in Test set...")
    check_file_paths(test_df, INPUT_DIR)


if __name__ == "__main__":
    try:
        # Generate
        t_path, v_path, te_path = generate_metadata()

        # Validate
        validate_metadata(t_path, v_path, te_path)

        print("\nMetadata generation and validation completed successfully.")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        # Re-raise to ensure the script fails explicitly
        raise e
