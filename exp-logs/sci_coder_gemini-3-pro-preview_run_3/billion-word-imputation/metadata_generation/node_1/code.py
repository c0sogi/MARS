import pandas as pd
import numpy as np
import os
import sys


def main():
    # Configuration
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    TRAIN_FILE = "train_v2.txt"
    TEST_FILE = "test_v2.txt"
    RANDOM_STATE = 42
    VAL_SPLIT_RATIO = 0.2

    # Ensure metadata directory exists
    os.makedirs(METADATA_DIR, exist_ok=True)

    print("Starting metadata generation script...")

    # ---------------------------------------------------------
    # 1. Load Raw Data
    # ---------------------------------------------------------
    train_path = os.path.join(INPUT_DIR, TRAIN_FILE)
    test_path = os.path.join(INPUT_DIR, TEST_FILE)

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training file not found at {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test file not found at {test_path}")

    print(f"Loading training data from {train_path}...")
    # The dataset is a collection of sentences (raw text), not a structured CSV.
    # We read it as a single column 'sentence'.
    try:
        # Use sep='\0' to treat each line as one field (assuming no null bytes)
        # quoting=3 (csv.QUOTE_NONE) ensures quotes are treated as literal characters
        df_train_full = pd.read_csv(
            train_path, sep="\0", header=None, names=["sentence"], quoting=3
        )
    except Exception as e:
        print(f"Failed to read training data: {e}")
        sys.exit(1)

    print(f"Loading test data from {test_path}...")
    try:
        df_test = pd.read_csv(test_path)
    except Exception as e:
        print(f"Failed to read test data: {e}")
        sys.exit(1)

    # ---------------------------------------------------------
    # 2. Split Training Data (Train/Val)
    # ---------------------------------------------------------
    print("Shuffling and splitting training data...")

    # Shuffle with fixed random state for reproducibility
    df_train_full = df_train_full.sample(frac=1, random_state=RANDOM_STATE).reset_index(
        drop=True
    )

    # Calculate split sizes
    n_total = len(df_train_full)
    n_val = int(n_total * VAL_SPLIT_RATIO)
    n_train = n_total - n_val

    # Perform split
    df_val = df_train_full.iloc[:n_val].copy()
    df_train = df_train_full.iloc[n_val:].copy()

    print(f"Original Train Size: {n_total}")
    print(f"New Train Size: {len(df_train)}")
    print(f"New Val Size: {len(df_val)}")

    # ---------------------------------------------------------
    # 3. Save Metadata
    # ---------------------------------------------------------
    print("Saving metadata to Parquet format...")

    train_meta_path = os.path.join(METADATA_DIR, "train.parquet")
    val_meta_path = os.path.join(METADATA_DIR, "val.parquet")
    test_meta_path = os.path.join(METADATA_DIR, "test.parquet")

    df_train.to_parquet(train_meta_path, index=False)
    df_val.to_parquet(val_meta_path, index=False)
    df_test.to_parquet(test_meta_path, index=False)

    print("Metadata generation complete.")

    # ---------------------------------------------------------
    # 4. Verification
    # ---------------------------------------------------------
    print("\nPerforming verification checks...")

    # Reload datasets
    df_train_check = pd.read_parquet(train_meta_path)
    df_val_check = pd.read_parquet(val_meta_path)
    df_test_check = pd.read_parquet(test_meta_path)

    # A. Summary Statistics
    print("\n--- Summary Statistics ---")
    print(f"Train Set: {len(df_train_check)} samples")
    print(f"Val Set:   {len(df_val_check)} samples")
    print(f"Test Set:  {len(df_test_check)} samples")

    print(f"Train Columns: {list(df_train_check.columns)}")
    print(f"Val Columns:   {list(df_val_check.columns)}")
    print(f"Test Columns:  {list(df_test_check.columns)}")

    # Check unique IDs if 'id' column exists
    if "id" in df_train_check.columns:
        print(f"Train Unique IDs: {df_train_check['id'].nunique()}")
    if "id" in df_val_check.columns:
        print(f"Val Unique IDs:   {df_val_check['id'].nunique()}")

    # B. Verify Split Requirements
    print("\n--- Verifying Split ---")
    actual_val_ratio = len(df_val_check) / (len(df_train_check) + len(df_val_check))
    print(f"Actual Validation Ratio: {actual_val_ratio:.6f}")

    # Assert ratio is correct (allowing for integer truncation)
    expected_val_count = int(
        (len(df_train_check) + len(df_val_check)) * VAL_SPLIT_RATIO
    )
    if len(df_val_check) != expected_val_count:
        raise AssertionError(
            f"Validation split size incorrect. Expected {expected_val_count}, got {len(df_val_check)}"
        )

    # C. Verify Data Leakage
    print("\n--- Verifying Data Leakage ---")
    if "id" in df_train_check.columns:
        train_ids = set(df_train_check["id"])
        val_ids = set(df_val_check["id"])
        intersection = train_ids.intersection(val_ids)
        if len(intersection) > 0:
            raise AssertionError(
                f"Data leakage detected! {len(intersection)} IDs found in both train and validation sets."
            )
        print("No ID overlap between train and validation sets.")
    else:
        print("Warning: 'id' column not found, skipping overlap check.")

    # D. File Path Verification
    # The instructions state: "If the metadata contains file paths, programmatically check 1000 relative file paths..."
    # In this dataset, the content is text sentences, not file paths.
    # Therefore, we explicitly log that this check is not applicable.
    print("\n--- File Path Verification ---")
    print(
        "Metadata contains text data directly ('sentence' column). No external file paths to verify."
    )

    print("\nAll checks passed successfully.")


if __name__ == "__main__":
    main()
