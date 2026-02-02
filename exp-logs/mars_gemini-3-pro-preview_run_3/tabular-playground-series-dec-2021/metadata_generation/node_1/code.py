import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sys


def main():
    # 1. Setup directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    os.makedirs(METADATA_DIR, exist_ok=True)

    print("Starting metadata generation...")

    # 2. Load raw data
    train_path = os.path.join(INPUT_DIR, "train.csv")
    test_path = os.path.join(INPUT_DIR, "test.csv")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"train.csv not found at {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"test.csv not found at {test_path}")

    # Using pandas to load. For 3.6M rows, this fits easily in 220GB RAM.
    print(f"Loading {train_path}...")
    df_train_full = pd.read_csv(train_path)
    print(f"Loading {test_path}...")
    df_test = pd.read_csv(test_path)

    target_col = "Cover_Type"
    id_col = "Id"

    # Check if target column exists
    if target_col not in df_train_full.columns:
        raise ValueError(f"Target column '{target_col}' not found in training data.")

    # 3. Create Validation Split
    # Requirements: 80:20, Random Shuffle, Fixed State 42, Stratified
    print("Splitting data into train and validation sets...")

    X = df_train_full
    y = df_train_full[target_col]

    # Filter out classes with fewer than 2 samples to enable stratified split
    class_counts = y.value_counts()
    valid_classes = class_counts[class_counts >= 2].index
    mask = y.isin(valid_classes)
    X = X[mask]
    y = y[mask]

    df_train, df_val = train_test_split(
        X, test_size=0.2, random_state=42, shuffle=True, stratify=y
    )

    # 4. Save Metadata
    # We save the actual data subsets as parquet for efficient loading by downstream tasks.
    # This serves as the "metadata" pointing to the specific rows for each set.
    train_meta_path = os.path.join(METADATA_DIR, "train.parquet")
    val_meta_path = os.path.join(METADATA_DIR, "val.parquet")
    test_meta_path = os.path.join(METADATA_DIR, "test.parquet")

    print(f"Saving metadata to {METADATA_DIR}...")
    df_train.to_parquet(train_meta_path, index=False)
    df_val.to_parquet(val_meta_path, index=False)
    df_test.to_parquet(test_meta_path, index=False)

    print("Metadata generation complete.")

    # ==========================================
    # Verification Steps
    # ==========================================
    print("\nRunning verification checks...")

    # 1. Load datasets using new metadata
    df_train_loaded = pd.read_parquet(train_meta_path)
    df_val_loaded = pd.read_parquet(val_meta_path)
    df_test_loaded = pd.read_parquet(test_meta_path)

    # 2. Print summary statistics
    print("\nSummary Statistics:")
    print("-" * 30)

    datasets = {
        "Train": df_train_loaded,
        "Validation": df_val_loaded,
        "Test": df_test_loaded,
    }

    for name, df in datasets.items():
        print(f"Dataset: {name}")
        print(f"  Shape: {df.shape}")
        if target_col in df.columns:
            dist = df[target_col].value_counts(normalize=True).sort_index()
            print(f"  Class Distribution (first 5): \n{dist.head()}")
            print(f"  Unique Classes: {df[target_col].nunique()}")
        if id_col in df.columns:
            print(f"  Unique IDs: {df[id_col].nunique()}")
        print("-" * 30)

    # 3. Check File Paths (if applicable)
    # The requirement is: "If the metadata contains file paths, programmatically check..."
    # We scan columns for string types that look like paths.
    # In this specific dataset (Cover Type), it is tabular, so likely no paths.
    # However, we implement the logic to satisfy the requirement generically.

    def check_paths(df, name):
        # Heuristic: check string columns containing '/' or starting with '.'
        # Since we loaded parquet, types are preserved.
        path_cols = []
        for col in df.select_dtypes(include=["object", "string"]).columns:
            # Check a sample
            sample = df[col].dropna().iloc[0] if not df[col].dropna().empty else ""
            if isinstance(sample, str) and ("/" in sample or sample.startswith(".")):
                path_cols.append(col)

        if not path_cols:
            print(
                f"No file path columns detected in {name} metadata. Skipping path check."
            )
            return

        print(f"Checking file paths in {name} (Columns: {path_cols})...")
        for col in path_cols:
            # Select 1000 random samples
            sample_paths = (
                df[col].sample(n=min(1000, len(df)), random_state=42).tolist()
            )
            missing_count = 0
            missing_samples = []

            for p in sample_paths:
                # Paths must be relative to ./input
                full_path = os.path.join(INPUT_DIR, str(p))
                # Handle case where path might already include ./input or be relative
                # The requirement says "relative to the ./input directory".
                # So if p is "image.jpg", we check "./input/image.jpg".
                if not os.path.exists(full_path):
                    missing_count += 1
                    if len(missing_samples) < 5:
                        missing_samples.append(full_path)

            missing_ratio = missing_count / len(sample_paths)
            print(f"  Column '{col}': Missing Ratio = {missing_ratio:.4f}")

            if missing_ratio > 0.5:
                print("  Sample missing paths:", missing_samples)
                raise FileNotFoundError(
                    f"More than 50% of files missing in column {col} of {name} dataset."
                )

    check_paths(df_train_loaded, "Train")
    check_paths(df_val_loaded, "Validation")
    check_paths(df_test_loaded, "Test")

    # 4. Verify Validation Set Requirements
    print("\nVerifying validation split requirements...")

    # Assert stratification
    train_dist = df_train_loaded[target_col].value_counts(normalize=True).sort_index()
    val_dist = df_val_loaded[target_col].value_counts(normalize=True).sort_index()

    # Align indexes to ensure we compare same classes (though they should be identical in stratified split)
    all_classes = sorted(list(set(train_dist.index) | set(val_dist.index)))
    train_dist = train_dist.reindex(all_classes, fill_value=0)
    val_dist = val_dist.reindex(all_classes, fill_value=0)

    diff = (train_dist - val_dist).abs().max()
    print(f"Max difference in class proportions between Train and Val: {diff:.6f}")

    # Tolerance: A small value. With 3.6M rows, stratification should be very good.
    if diff > 0.01:  # 1% tolerance
        raise AssertionError(
            "Stratification failed: Class distribution differs significantly between train and validation sets."
        )

    # Assert No Leakage (ID overlap)
    train_ids = set(df_train_loaded[id_col])
    val_ids = set(df_val_loaded[id_col])

    overlap = train_ids.intersection(val_ids)
    if len(overlap) > 0:
        raise AssertionError(
            f"Data Leakage detected: {len(overlap)} IDs found in both train and validation sets."
        )

    print("Verification passed successfully.")


if __name__ == "__main__":
    main()
