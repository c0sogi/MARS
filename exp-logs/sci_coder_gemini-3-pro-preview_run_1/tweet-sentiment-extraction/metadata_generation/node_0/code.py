import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split


def generate_metadata():
    # Define directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Ensure metadata directory exists
    os.makedirs(METADATA_DIR, exist_ok=True)

    # Load raw datasets
    train_path = os.path.join(INPUT_DIR, "train.csv")
    test_path = os.path.join(INPUT_DIR, "test.csv")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train file not found at {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test file not found at {test_path}")

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    # Preprocessing: Drop rows with NaN values in critical columns
    # The dataset is known to have a few rows with missing text
    initial_train_size = len(df_train)
    df_train = df_train.dropna(subset=["text", "selected_text", "sentiment"])
    dropped_count = initial_train_size - len(df_train)
    if dropped_count > 0:
        print(f"Dropped {dropped_count} rows with missing values from training data.")

    # Create Validation Split
    # Requirements: 80:20 split, Random State 42, Stratified by sentiment
    train_df, val_df = train_test_split(
        df_train, test_size=0.2, random_state=42, stratify=df_train["sentiment"]
    )

    # Save metadata files
    train_meta_path = os.path.join(METADATA_DIR, "train_metadata.csv")
    val_meta_path = os.path.join(METADATA_DIR, "validation_metadata.csv")
    test_meta_path = os.path.join(METADATA_DIR, "test_metadata.csv")

    train_df.to_csv(train_meta_path, index=False)
    val_df.to_csv(val_meta_path, index=False)
    df_test.to_csv(test_meta_path, index=False)

    print("Metadata files generated successfully.")

    return train_meta_path, val_meta_path, test_meta_path


def check_metadata(train_path, val_path, test_path):
    print("\nPerforming Validation Checks...")

    # Load datasets
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    # 1. Summary Statistics
    print("\n--- Summary Statistics ---")
    print(f"Train Set Shape: {df_train.shape}")
    print(f"Validation Set Shape: {df_val.shape}")
    print(f"Test Set Shape: {df_test.shape}")

    print("\nTrain Sentiment Distribution:")
    train_dist = df_train["sentiment"].value_counts(normalize=True).sort_index()
    print(train_dist)

    print("\nValidation Sentiment Distribution:")
    val_dist = df_val["sentiment"].value_counts(normalize=True).sort_index()
    print(val_dist)

    # 2. File Path Check
    # This dataset contains text directly in the CSV and does not reference external files.
    # Therefore, the "missing file ratio" check is not applicable.
    print(
        "\nFile Path Check: Skipped (Dataset contains text content, no external file paths)."
    )

    # 3. Verify Validation Set Requirements
    print("\nVerifying Split Requirements...")

    # Check Split Ratio (80:20)
    total_samples = len(df_train) + len(df_val)
    val_ratio = len(df_val) / total_samples
    print(f"Actual Validation Ratio: {val_ratio:.5f}")

    # Assert ratio is within a reasonable margin of error (e.g., +/- 1%)
    if not (0.19 <= val_ratio <= 0.21):
        raise AssertionError(
            f"Validation split ratio {val_ratio:.4f} deviates significantly from 0.2"
        )

    # Check Stratification
    # We compare the normalized value counts of the 'sentiment' column
    # The difference should be minimal
    diff = (train_dist - val_dist).abs().max()
    print(f"Max difference in sentiment class proportions: {diff:.5f}")

    if diff > 0.015:  # Allow 1.5% tolerance for discrete sampling differences
        raise AssertionError(
            "Stratification failed: Class distributions differ significantly."
        )

    print("\nAll validation checks passed successfully.")


if __name__ == "__main__":
    try:
        t_path, v_path, te_path = generate_metadata()
        check_metadata(t_path, v_path, te_path)
    except Exception as e:
        print(f"\nERROR: {e}")
        exit(1)
