import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import random


def run_metadata_generation():
    # Configuration
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    TRAIN_FILE = "train_v2.txt"
    TEST_FILE = "test_v2.txt"
    RANDOM_STATE = 42

    # Create metadata directory
    os.makedirs(METADATA_DIR, exist_ok=True)

    print("Starting metadata generation...")

    # ---------------------------------------------------------
    # 1. Process Test Data
    # ---------------------------------------------------------
    print(f"Processing {TEST_FILE}...")
    test_path = os.path.join(INPUT_DIR, TEST_FILE)

    if not os.path.exists(test_path):
        raise FileNotFoundError(f"{test_path} not found.")

    # Test file is described as having a header: "id","sentence"
    # We use engine='python' to handle potential complex quoting robustly, though 'c' is faster
    df_test = pd.read_csv(test_path)

    # Add relative file path
    df_test["file_path"] = TEST_FILE

    # Save to parquet
    test_meta_path = os.path.join(METADATA_DIR, "test.parquet")
    df_test.to_parquet(test_meta_path, index=False)
    print(f"Saved test metadata to {test_meta_path} with {len(df_test)} samples.")

    # ---------------------------------------------------------
    # 2. Process Train Data
    # ---------------------------------------------------------
    print(f"Processing {TRAIN_FILE}...")
    train_path = os.path.join(INPUT_DIR, TRAIN_FILE)

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"{train_path} not found.")

    # Heuristic to check format (CSV vs Raw Text)
    # The prompt implies train might be just a collection of sentences
    with open(train_path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()

    is_csv = False
    if first_line.startswith('"id","sentence"') or first_line.startswith(
        'id,"sentence"'
    ):
        is_csv = True

    if is_csv:
        print("Detected CSV format for training data.")
        df_train_full = pd.read_csv(train_path)
    else:
        print("Detected raw text format for training data. Reading lines...")
        # Read all lines. Given 220GB RAM, 30M lines fits in memory.
        with open(train_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f]

        # Filter empty lines
        lines = [l for l in lines if l]

        df_train_full = pd.DataFrame({"sentence": lines})
        # Generate IDs
        df_train_full["id"] = df_train_full.index

    df_train_full["file_path"] = TRAIN_FILE

    print(f"Total training samples loaded: {len(df_train_full)}")

    # ---------------------------------------------------------
    # 3. Create Validation Split
    # ---------------------------------------------------------
    print("Creating 80:20 Train/Validation split...")

    # Since this is a sentence completion task with no explicit class labels provided
    # in the raw text, we perform a random split. Stratified sampling is not applicable
    # without pre-computed discrete labels.
    df_train, df_val = train_test_split(
        df_train_full, test_size=0.2, random_state=RANDOM_STATE, shuffle=True
    )

    # Save to parquet
    train_meta_path = os.path.join(METADATA_DIR, "train.parquet")
    val_meta_path = os.path.join(METADATA_DIR, "val.parquet")

    df_train.to_parquet(train_meta_path, index=False)
    df_val.to_parquet(val_meta_path, index=False)

    print(f"Saved train metadata to {train_meta_path} ({len(df_train)} samples)")
    print(f"Saved val metadata to {val_meta_path} ({len(df_val)} samples)")

    # Free memory
    del df_train_full, df_train, df_val, df_test

    # ---------------------------------------------------------
    # 4. Verification and Checks
    # ---------------------------------------------------------
    print("\nRunning verification checks...")

    # Reload datasets
    df_train_check = pd.read_parquet(train_meta_path)
    df_val_check = pd.read_parquet(val_meta_path)
    df_test_check = pd.read_parquet(test_meta_path)

    # 4a. Summary Statistics
    print("--- Summary Statistics ---")
    print(
        f"Train Set: {df_train_check.shape[0]} rows, Columns: {list(df_train_check.columns)}"
    )
    print(
        f"Val Set:   {df_val_check.shape[0]} rows, Columns: {list(df_val_check.columns)}"
    )
    print(
        f"Test Set:  {df_test_check.shape[0]} rows, Columns: {list(df_test_check.columns)}"
    )

    # 4b. File Path Verification
    datasets = {"Train": df_train_check, "Val": df_val_check, "Test": df_test_check}

    for name, df in datasets.items():
        # Sample 1000 paths (or all if less than 1000)
        n_sample = min(1000, len(df))
        sample_paths = (
            df["file_path"].sample(n=n_sample, random_state=RANDOM_STATE).tolist()
        )

        missing_count = 0
        missing_examples = []

        for rel_path in sample_paths:
            full_path = os.path.join(INPUT_DIR, rel_path)
            if not os.path.exists(full_path):
                missing_count += 1
                if len(missing_examples) < 5:
                    missing_examples.append(rel_path)

        missing_ratio = missing_count / n_sample
        print(f"[{name}] Missing file ratio: {missing_ratio:.4f}")

        if missing_ratio > 0.5:
            print(f"Sample missing paths: {missing_examples}")
            raise FileNotFoundError(
                f"Missing file ratio ({missing_ratio}) exceeded threshold for {name} dataset."
            )

    # 4c. Split Verification
    total_train = len(df_train_check)
    total_val = len(df_val_check)
    total_samples = total_train + total_val

    actual_val_ratio = total_val / total_samples
    print(f"Actual Validation Ratio: {actual_val_ratio:.5f}")

    # Assert ratio is close to 0.2 (allowing for small integer division variance)
    if not (0.199 <= actual_val_ratio <= 0.201):
        raise AssertionError(
            f"Validation split ratio {actual_val_ratio} is not approximately 0.2"
        )

    print("\nMetadata generation and verification completed successfully.")


if __name__ == "__main__":
    run_metadata_generation()
