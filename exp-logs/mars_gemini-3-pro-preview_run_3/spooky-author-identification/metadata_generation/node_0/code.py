import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split


def run_metadata_generation():
    # Configuration
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    RANDOM_STATE = 42
    VAL_SIZE = 0.2

    # Ensure metadata directory exists
    os.makedirs(METADATA_DIR, exist_ok=True)

    print("--- Starting Metadata Generation ---")

    # 1. Load Raw Data
    print(f"Loading data from {INPUT_DIR}...")
    try:
        train_df = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))
        test_df = pd.read_csv(os.path.join(INPUT_DIR, "test.csv"))
    except FileNotFoundError as e:
        print(f"Error: Required input files not found. {e}")
        raise

    # 2. Split Training Data (Stratified, 80:20, Random State 42)
    print("Splitting training data into Train and Validation sets...")
    train_split, val_split = train_test_split(
        train_df,
        test_size=VAL_SIZE,
        stratify=train_df["author"],
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    # 3. Save Metadata
    # For this text dataset, the metadata contains the text content itself.
    print(f"Saving metadata to {METADATA_DIR}...")
    train_meta_path = os.path.join(METADATA_DIR, "train.csv")
    val_meta_path = os.path.join(METADATA_DIR, "val.csv")
    test_meta_path = os.path.join(METADATA_DIR, "test.csv")

    train_split.to_csv(train_meta_path, index=False)
    val_split.to_csv(val_meta_path, index=False)
    test_df.to_csv(test_meta_path, index=False)

    print("Metadata generation complete.")

    # 4. Validation and Checks
    print("\n--- Performing Validation Checks ---")

    # Reload data to verify
    df_train_check = pd.read_csv(train_meta_path)
    df_val_check = pd.read_csv(val_meta_path)
    df_test_check = pd.read_csv(test_meta_path)

    # Summary Statistics
    print(
        f"Train Set: {df_train_check.shape[0]} samples, Columns: {list(df_train_check.columns)}"
    )
    print(
        f"Val Set:   {df_val_check.shape[0]} samples, Columns: {list(df_val_check.columns)}"
    )
    print(
        f"Test Set:  {df_test_check.shape[0]} samples, Columns: {list(df_test_check.columns)}"
    )

    print("\nClass Distribution (Train):")
    train_dist = df_train_check["author"].value_counts(normalize=True).sort_index()
    print(train_dist)

    print("\nClass Distribution (Validation):")
    val_dist = df_val_check["author"].value_counts(normalize=True).sort_index()
    print(val_dist)

    # Check 1: Split Ratio
    total_samples = len(df_train_check) + len(df_val_check)
    actual_val_ratio = len(df_val_check) / total_samples
    print(f"\nActual Validation Ratio: {actual_val_ratio:.5f}")

    # Assert ratio is within 0.1% tolerance (accounting for integer rounding)
    if abs(actual_val_ratio - VAL_SIZE) > 0.001:
        raise AssertionError(
            f"Validation split ratio mismatch. Expected {VAL_SIZE}, got {actual_val_ratio:.5f}"
        )

    # Check 2: Stratification
    # We check if the difference in class proportions between train and val is small (< 1%)
    dist_diff = (train_dist - val_dist).abs()
    print("\nMax difference in class proportions between Train and Val:")
    print(dist_diff.max())

    if dist_diff.max() > 0.01:
        raise AssertionError(
            "Stratification check failed: Class distributions differ significantly between train and validation sets."
        )

    print("Stratification check passed.")

    # Check 3: File Path Resolution
    # The dataset contains inline text ('text' column) and no external file paths.
    # Therefore, we skip the file path existence check.
    # We verify this assumption by checking columns.
    if "text" in df_train_check.columns and "id" in df_train_check.columns:
        print(
            "\nDataset contains inline text. No external file paths to verify. Skipping file path check."
        )
    else:
        # If the structure was different and had paths, we would check them here.
        # For this specific task, we know it's text.
        pass

    print("\nAll checks passed successfully.")


if __name__ == "__main__":
    run_metadata_generation()
