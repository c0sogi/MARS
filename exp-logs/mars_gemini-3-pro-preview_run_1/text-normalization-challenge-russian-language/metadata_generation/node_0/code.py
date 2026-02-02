import os
import pandas as pd
import numpy as np
import sys


def main():
    # Configuration
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    TRAIN_FILE = "ru_train.csv"
    TEST_FILE = "ru_test_2.csv"
    RANDOM_STATE = 42
    VAL_SIZE = 0.2

    # Ensure metadata directory exists
    os.makedirs(METADATA_DIR, exist_ok=True)

    print("Loading raw data...")
    train_path = os.path.join(INPUT_DIR, TRAIN_FILE)
    test_path = os.path.join(INPUT_DIR, TEST_FILE)

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training file not found at {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test file not found at {test_path}")

    # Load training data
    # specifying dtypes can help with memory and speed, but defaults are usually fine for this size on this hardware
    df_train = pd.read_csv(train_path)
    print(f"Loaded train data: {df_train.shape}")

    # Load test data
    df_test = pd.read_csv(test_path)
    print(f"Loaded test data: {df_test.shape}")

    # --- Split Train/Val ---
    # Requirement: Group Sampling by sentence_id
    print("Performing group split by sentence_id...")

    if "sentence_id" not in df_train.columns:
        raise ValueError("Column 'sentence_id' not found in training data.")

    unique_sentences = df_train["sentence_id"].unique()
    print(f"Total unique sentences: {len(unique_sentences)}")

    # Shuffle unique sentences
    np.random.seed(RANDOM_STATE)
    np.random.shuffle(unique_sentences)

    # Determine split index
    n_val = int(len(unique_sentences) * VAL_SIZE)
    val_sentences = set(unique_sentences[:n_val])
    train_sentences = set(unique_sentences[n_val:])

    # Create masks
    # Using isin is efficient enough here
    train_mask = df_train["sentence_id"].isin(train_sentences)
    val_mask = df_train["sentence_id"].isin(val_sentences)

    df_train_split = df_train[train_mask].copy()
    df_val_split = df_train[val_mask].copy()

    print(f"Train split shape: {df_train_split.shape}")
    print(f"Val split shape: {df_val_split.shape}")

    # --- Save Metadata ---
    print("Saving metadata files...")
    train_meta_path = os.path.join(METADATA_DIR, "train.csv")
    val_meta_path = os.path.join(METADATA_DIR, "val.csv")
    test_meta_path = os.path.join(METADATA_DIR, "test.csv")

    df_train_split.to_csv(train_meta_path, index=False)
    df_val_split.to_csv(val_meta_path, index=False)
    df_test.to_csv(test_meta_path, index=False)

    # --- Verification ---
    print("\n--- Verifying Metadata ---")

    # Reload datasets
    df_train_check = pd.read_csv(train_meta_path)
    df_val_check = pd.read_csv(val_meta_path)
    df_test_check = pd.read_csv(test_meta_path)

    # 1. Summary Statistics
    print("\nSummary Statistics:")
    print(f"Train samples: {len(df_train_check)}")
    print(f"Val samples:   {len(df_val_check)}")
    print(f"Test samples:  {len(df_test_check)}")

    if "class" in df_train_check.columns:
        print("\nTrain Class Distribution:")
        print(df_train_check["class"].value_counts(normalize=True).head())
        print("\nVal Class Distribution:")
        print(df_val_check["class"].value_counts(normalize=True).head())

    # 2. Check File Paths
    # The dataset is text-based in CSVs, so there are no external file paths (like images/audio) to check.
    # However, we will implement the logic as requested, checking if any column looks like a relative file path.
    # In this specific dataset, columns are IDs, text, classes. None are file paths.
    # We will explicitly state this.
    print("\nChecking file paths...")
    potential_path_cols = [
        col
        for col in df_train_check.columns
        if "path" in col.lower() or "file" in col.lower()
    ]
    if potential_path_cols:
        # If there were path columns, we would check them here.
        pass
    else:
        print(
            "No explicit file path columns detected in metadata. Skipping file existence check."
        )

    # 3. Verify Validation Split Requirements
    print("\nVerifying split integrity...")

    # Assert group split was successful (no sentence leakage)
    train_sent_ids = set(df_train_check["sentence_id"].unique())
    val_sent_ids = set(df_val_check["sentence_id"].unique())

    intersection = train_sent_ids.intersection(val_sent_ids)
    if len(intersection) > 0:
        raise AssertionError(
            f"Data Leakage Detected! {len(intersection)} sentence_ids found in both train and val."
        )

    # Assert ratio is roughly correct (in terms of groups)
    n_train_groups = len(train_sent_ids)
    n_val_groups = len(val_sent_ids)
    total_groups = n_train_groups + n_val_groups
    actual_val_ratio = n_val_groups / total_groups

    print(f"Group Split Ratio (Val): {actual_val_ratio:.4f} (Target: {VAL_SIZE})")

    # Allow small deviation due to rounding
    if not (0.19 < actual_val_ratio < 0.21):
        print(
            f"Warning: Validation ratio {actual_val_ratio} deviates from target {VAL_SIZE}"
        )
        # Note: We don't raise error here because exact ratio depends on integer rounding of groups,
        # but for large N it should be close.

    print("\nVerification passed successfully.")


if __name__ == "__main__":
    main()
