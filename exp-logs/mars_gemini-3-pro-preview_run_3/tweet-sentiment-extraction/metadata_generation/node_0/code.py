import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

# Configuration
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
RANDOM_STATE = 42
VAL_SIZE = 0.2


def generate_metadata():
    # Ensure metadata directory exists
    os.makedirs(METADATA_DIR, exist_ok=True)

    # 1. Load Data
    train_path = os.path.join(INPUT_DIR, "train.csv")
    test_path = os.path.join(INPUT_DIR, "test.csv")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError("Input files not found.")

    df_train_full = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    # 2. Preprocessing
    # Drop rows where critical columns are NaN
    # For training, we need text, selected_text, and sentiment
    initial_train_len = len(df_train_full)
    df_train_full = df_train_full.dropna(subset=["text", "selected_text", "sentiment"])
    dropped_count = initial_train_len - len(df_train_full)
    if dropped_count > 0:
        print(f"Dropped {dropped_count} rows with missing values from training set.")

    # 3. Split Training Data (Stratified)
    # We stratify by 'sentiment' to ensure balanced classes in validation
    train_df, val_df = train_test_split(
        df_train_full,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=df_train_full["sentiment"],
    )

    # 4. Save Metadata
    # In this task, the CSVs themselves act as the metadata/dataset
    train_meta_path = os.path.join(METADATA_DIR, "train.csv")
    val_meta_path = os.path.join(METADATA_DIR, "val.csv")
    test_meta_path = os.path.join(METADATA_DIR, "test.csv")

    train_df.to_csv(train_meta_path, index=False)
    val_df.to_csv(val_meta_path, index=False)
    df_test.to_csv(test_meta_path, index=False)

    print(f"Metadata saved to {METADATA_DIR}")

    return train_meta_path, val_meta_path, test_meta_path


def verify_metadata(train_path, val_path, test_path):
    print("\n--- Verifying Metadata ---")

    # Load datasets
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    datasets = {"Train": df_train, "Validation": df_val, "Test": df_test}

    # 1. Print Summary Statistics
    for name, df in datasets.items():
        print(f"\nDataset: {name}")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")

        if "sentiment" in df.columns:
            print("  Sentiment Distribution:")
            print(df["sentiment"].value_counts(normalize=True))

        if "textID" in df.columns:
            unique_ids = df["textID"].nunique()
            print(f"  Unique Users/IDs: {unique_ids}")

    # 2. File Path Check
    # This dataset consists of text inside CSVs, not paths to external files.
    # Therefore, we skip the path resolution check.
    print(
        "\nFile Path Check: No external file path columns detected. Skipping resolution check."
    )

    # 3. Verify Validation Split Requirements
    print("\nVerifying Validation Split...")

    # Check Stratification
    train_dist = df_train["sentiment"].value_counts(normalize=True).sort_index()
    val_dist = df_val["sentiment"].value_counts(normalize=True).sort_index()

    diff = (train_dist - val_dist).abs().max()
    print(f"Max difference in sentiment proportions between Train and Val: {diff:.6f}")

    # Assert stratification is successful (allow very small margin for discrete count differences)
    if diff > 0.015:
        raise AssertionError(
            "Stratification failed! Validation distribution deviates significantly from training."
        )

    # Check for Data Leakage (Overlap)
    train_ids = set(df_train["textID"])
    val_ids = set(df_val["textID"])
    overlap = train_ids.intersection(val_ids)

    if overlap:
        raise AssertionError(
            f"Data Leakage detected! {len(overlap)} IDs found in both train and validation sets."
        )

    print("Verification passed: Split is stratified and disjoint.")


if __name__ == "__main__":
    try:
        t_path, v_path, te_path = generate_metadata()
        verify_metadata(t_path, v_path, te_path)
    except Exception as e:
        print(f"\nERROR: {e}")
        raise e
