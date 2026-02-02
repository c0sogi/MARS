import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


def main():
    # Configuration
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    RANDOM_STATE = 42
    VAL_SIZE = 0.2

    # Create metadata directory
    os.makedirs(METADATA_DIR, exist_ok=True)

    print("Loading raw data...")

    # Locate files
    train_path = os.path.join(INPUT_DIR, "train.json")
    if not os.path.exists(train_path):
        train_path = os.path.join(INPUT_DIR, "train", "train.json")

    test_path = os.path.join(INPUT_DIR, "test.json")
    if not os.path.exists(test_path):
        test_path = os.path.join(INPUT_DIR, "test", "test.json")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(
            "Could not locate train.json or test.json in input directory."
        )

    # Load JSON data
    train_df = pd.read_json(train_path)
    test_df = pd.read_json(test_path)

    # Add source file column (relative path) for verification requirement
    # We store the path relative to ./input
    train_rel_path = os.path.relpath(train_path, INPUT_DIR)
    test_rel_path = os.path.relpath(test_path, INPUT_DIR)

    train_df["source_file"] = train_rel_path
    test_df["source_file"] = test_rel_path

    # Ensure target is present in train
    target_col = "requester_received_pizza"
    if target_col not in train_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in training data.")

    # Convert boolean target to int for consistency
    train_df[target_col] = train_df[target_col].astype(int)

    print(f"Raw Train shape: {train_df.shape}")
    print(f"Raw Test shape: {test_df.shape}")

    # Stratified Split
    print("Performing stratified split...")
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=VAL_SIZE, random_state=RANDOM_STATE
    )

    # We split based on the target
    split_indices = list(splitter.split(train_df, train_df[target_col]))
    train_idx, val_idx = split_indices[0]

    new_train_df = train_df.iloc[train_idx].copy()
    new_val_df = train_df.iloc[val_idx].copy()

    # Save to metadata as Parquet (efficient for text and tabular data)
    print("Saving metadata files...")
    train_meta_path = os.path.join(METADATA_DIR, "train.parquet")
    val_meta_path = os.path.join(METADATA_DIR, "val.parquet")
    test_meta_path = os.path.join(METADATA_DIR, "test.parquet")

    new_train_df.to_parquet(train_meta_path, index=False)
    new_val_df.to_parquet(val_meta_path, index=False)
    test_df.to_parquet(test_meta_path, index=False)

    print("Metadata generation complete.")

    # ---------------------------------------------------------
    # Verification Step
    # ---------------------------------------------------------
    print("\nVerifying generated metadata...")

    # Load back the data
    loaded_train = pd.read_parquet(train_meta_path)
    loaded_val = pd.read_parquet(val_meta_path)
    loaded_test = pd.read_parquet(test_meta_path)

    # 1. Summary Statistics
    print("\n--- Summary Statistics ---")
    print(f"Train set: {len(loaded_train)} samples")
    print(f"Val set:   {len(loaded_val)} samples")
    print(f"Test set:  {len(loaded_test)} samples")

    train_dist = loaded_train[target_col].value_counts(normalize=True)
    val_dist = loaded_val[target_col].value_counts(normalize=True)

    print(f"\nTrain Class Distribution:\n{train_dist}")
    print(f"\nVal Class Distribution:\n{val_dist}")

    print(f"\nUnique users in Train: {loaded_train['requester_username'].nunique()}")
    print(f"Unique users in Val:   {loaded_val['requester_username'].nunique()}")

    # 2. File Path Verification
    # We check the 'source_file' column we added.
    print("\n--- Verifying File Paths ---")

    def verify_paths(df, name):
        if "source_file" not in df.columns:
            return

        paths = df["source_file"].sample(n=min(1000, len(df)), random_state=42).tolist()
        missing_count = 0
        missing_samples = []

        for p in paths:
            full_path = os.path.join(INPUT_DIR, p)
            if not os.path.exists(full_path):
                missing_count += 1
                if len(missing_samples) < 5:
                    missing_samples.append(p)

        ratio = missing_count / len(paths)
        print(f"Missing file ratio for {name}: {ratio:.4f}")

        if ratio > 0.5:
            print(f"Sample missing paths: {missing_samples}")
            raise FileNotFoundError(
                f"More than 50% of file paths in {name} are invalid."
            )

    verify_paths(loaded_train, "Train")
    verify_paths(loaded_val, "Val")
    verify_paths(loaded_test, "Test")

    # 3. Verify Stratification
    print("\n--- Verifying Stratification ---")
    train_pos_ratio = loaded_train[target_col].mean()
    val_pos_ratio = loaded_val[target_col].mean()

    print(f"Train Positive Ratio: {train_pos_ratio:.4f}")
    print(f"Val Positive Ratio:   {val_pos_ratio:.4f}")

    # Check if ratios are close (within 1%)
    diff = abs(train_pos_ratio - val_pos_ratio)
    if diff > 0.01:
        raise AssertionError(
            f"Stratification failed! Difference in positive class ratio is {diff:.4f}, expected < 0.01"
        )

    print("Stratification check passed.")

    # 4. Verify No Leakage
    train_ids = set(loaded_train["request_id"])
    val_ids = set(loaded_val["request_id"])
    intersection = train_ids.intersection(val_ids)

    if intersection:
        raise AssertionError(
            f"Data leakage detected! {len(intersection)} IDs found in both train and val."
        )

    print("Leakage check passed.")
    print("\nAll checks passed successfully.")


if __name__ == "__main__":
    main()
