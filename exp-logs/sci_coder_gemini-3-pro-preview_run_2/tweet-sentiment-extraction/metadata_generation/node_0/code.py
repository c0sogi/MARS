import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


def generate_metadata():
    # Define paths
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Create metadata directory
    os.makedirs(METADATA_DIR, exist_ok=True)

    # Load raw data
    train_path = os.path.join(INPUT_DIR, "train.csv")
    test_path = os.path.join(INPUT_DIR, "test.csv")

    print(f"Loading data from {INPUT_DIR}...")
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    # Preprocessing: Clean training data
    # Remove rows where text or selected_text is NaN
    initial_train_len = len(df_train)
    df_train = df_train.dropna(subset=["text", "selected_text", "sentiment"])
    dropped_count = initial_train_len - len(df_train)
    if dropped_count > 0:
        print(f"Dropped {dropped_count} rows with missing values from training data.")

    # Perform Stratified Split (80/20)
    # We stratify based on the 'sentiment' column to maintain class distribution
    print("Splitting training data into Train/Validation (80:20)...")
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    # Get indices
    train_idx, val_idx = next(splitter.split(df_train, df_train["sentiment"]))

    # Create dataframes
    train_split = df_train.iloc[train_idx].copy()
    val_split = df_train.iloc[val_idx].copy()

    # Save metadata files
    # Since the data is text, the metadata contains the content itself
    train_meta_path = os.path.join(METADATA_DIR, "train.csv")
    val_meta_path = os.path.join(METADATA_DIR, "val.csv")
    test_meta_path = os.path.join(METADATA_DIR, "test.csv")

    print(f"Saving metadata to {METADATA_DIR}...")
    train_split.to_csv(train_meta_path, index=False)
    val_split.to_csv(val_meta_path, index=False)
    df_test.to_csv(test_meta_path, index=False)

    return train_meta_path, val_meta_path, test_meta_path


def validate_metadata(train_path, val_path, test_path):
    print("\n--- Validating Metadata ---")

    # Load generated metadata
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    # 1. Print Summary Statistics
    print(f"Train Set: {df_train.shape[0]} samples")
    print(f"Val Set:   {df_val.shape[0]} samples")
    print(f"Test Set:  {df_test.shape[0]} samples")

    print("\nTrain Sentiment Distribution:")
    train_dist = df_train["sentiment"].value_counts(normalize=True)
    print(train_dist)

    print("\nValidation Sentiment Distribution:")
    val_dist = df_val["sentiment"].value_counts(normalize=True)
    print(val_dist)

    # 2. Verify Split Ratio
    total_train = len(df_train) + len(df_val)
    val_ratio = len(df_val) / total_train
    print(f"\nActual Validation Ratio: {val_ratio:.5f}")

    # Assert ratio is 0.2 (+/- small epsilon due to rounding/discrete sizes)
    if not (0.199 <= val_ratio <= 0.201):
        raise AssertionError(f"Validation split ratio {val_ratio} is not close to 0.2")

    # 3. Verify Stratification
    # Check if distributions match within a tolerance (e.g., 1%)
    print("\nVerifying Stratification...")
    for label in train_dist.index:
        train_prop = train_dist[label]
        val_prop = val_dist.get(label, 0)
        diff = abs(train_prop - val_prop)
        if diff > 0.01:  # 1% tolerance
            raise AssertionError(
                f"Stratification failed for class '{label}'. Train: {train_prop:.4f}, Val: {val_prop:.4f}, Diff: {diff:.4f}"
            )
    print("Stratification check passed.")

    # 4. File Path Check
    # The dataset consists of text strings within the CSV, not paths to external files.
    # Therefore, checking for "missing files" on the filesystem is not applicable to columns like 'text' or 'selected_text'.
    # We skip the file resolution check as there are no file path columns.
    print("No external file path columns detected. Skipping file resolution check.")


if __name__ == "__main__":
    try:
        # Generate
        t_path, v_path, te_path = generate_metadata()

        # Validate
        validate_metadata(t_path, v_path, te_path)

        print("\nMetadata generation and validation completed successfully.")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        exit(1)
