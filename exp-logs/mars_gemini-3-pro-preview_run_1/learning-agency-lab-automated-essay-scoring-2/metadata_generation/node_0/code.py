import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


def generate_metadata():
    # Define paths
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    TRAIN_PATH = os.path.join(INPUT_DIR, "train.csv")
    TEST_PATH = os.path.join(INPUT_DIR, "test.csv")

    # Create metadata directory
    os.makedirs(METADATA_DIR, exist_ok=True)

    # Load datasets
    print("Loading raw data...")
    # Using pandas to handle potential newlines in 'full_text' correctly
    df_train_full = pd.read_csv(TRAIN_PATH)
    df_test = pd.read_csv(TEST_PATH)

    print(f"Original Train shape: {df_train_full.shape}")
    print(f"Original Test shape: {df_test.shape}")

    # Perform Stratified Split
    # 80% Train, 20% Validation
    # Stratify by 'score'
    print("Splitting data...")
    if "score" not in df_train_full.columns:
        raise ValueError("Column 'score' not found in training data.")

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)

    split_indices = list(splitter.split(df_train_full, df_train_full["score"]))
    train_idx, val_idx = split_indices[0]

    df_train = df_train_full.iloc[train_idx].copy()
    df_val = df_train_full.iloc[val_idx].copy()

    # Save metadata
    # For NLP tasks where data is text in CSV, the CSV itself serves as metadata/data
    print("Saving metadata...")
    train_meta_path = os.path.join(METADATA_DIR, "train.csv")
    val_meta_path = os.path.join(METADATA_DIR, "val.csv")
    test_meta_path = os.path.join(METADATA_DIR, "test.csv")

    df_train.to_csv(train_meta_path, index=False)
    df_val.to_csv(val_meta_path, index=False)
    df_test.to_csv(test_meta_path, index=False)

    # Verification Step
    print("Verifying generated metadata...")

    # Reload data to ensure integrity
    df_train_check = pd.read_csv(train_meta_path)
    df_val_check = pd.read_csv(val_meta_path)
    df_test_check = pd.read_csv(test_meta_path)

    # 1. Summary Statistics
    print("\n=== Summary Statistics ===")
    print(f"Train Set: {len(df_train_check)} samples")
    print(f"Validation Set: {len(df_val_check)} samples")
    print(f"Test Set: {len(df_test_check)} samples")

    print("\nTrain Score Distribution:")
    print(df_train_check["score"].value_counts(normalize=True).sort_index())

    print("\nValidation Score Distribution:")
    print(df_val_check["score"].value_counts(normalize=True).sort_index())

    # 2. Check File Paths
    # The dataset consists of text within the CSVs. There are no external file paths
    # (e.g., images/audio) relative to input/ to check.

    # 3. Verify Validation Split Requirements
    print("\nVerifying split requirements...")

    # Check Split Ratio (80:20)
    total_train_val = len(df_train_check) + len(df_val_check)
    val_ratio = len(df_val_check) / total_train_val
    print(f"Actual Validation Ratio: {val_ratio:.4f}")

    # Assert ratio is approximately 0.20
    assert (
        0.19 < val_ratio < 0.21
    ), f"Validation ratio {val_ratio} is not approximately 0.20"

    # Check Stratification
    # We compare the normalized value counts of the score column
    train_dist = df_train_check["score"].value_counts(normalize=True).sort_index()
    val_dist = df_val_check["score"].value_counts(normalize=True).sort_index()

    # Calculate maximum absolute difference in class proportions
    all_classes = sorted(list(set(train_dist.index) | set(val_dist.index)))
    max_diff = 0.0
    for cls in all_classes:
        t_p = train_dist.get(cls, 0)
        v_p = val_dist.get(cls, 0)
        diff = abs(t_p - v_p)
        max_diff = max(max_diff, diff)

    print(f"Max class proportion difference between Train and Val: {max_diff:.4f}")

    # Assert stratification was successful (difference should be very small)
    assert (
        max_diff < 0.015
    ), f"Stratification failed. Max difference in class proportions: {max_diff}"

    print("\nMetadata generation and verification completed successfully.")


if __name__ == "__main__":
    generate_metadata()
