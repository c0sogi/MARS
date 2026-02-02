import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split


def main():
    # Define directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Ensure metadata directory exists
    os.makedirs(METADATA_DIR, exist_ok=True)

    print("Loading raw data...")
    train_path = os.path.join(INPUT_DIR, "train.csv")
    test_path = os.path.join(INPUT_DIR, "test.csv")

    # Load datasets
    # We assume standard CSV format with headers based on sample_submission info
    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
    except Exception as e:
        raise RuntimeError(f"Failed to read input CSV files: {e}")

    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    print(f"Train columns: {train_df.columns.tolist()}")

    # Identify label column
    label_col = "Insult"
    if label_col not in train_df.columns:
        # Fallback logic: if headers are missing, try to infer or raise error
        # Given the task description, we expect 'Insult' to be present.
        raise ValueError(f"Expected label column '{label_col}' not found in train.csv")

    # Split training data into train and validation
    print("Splitting data into Train (80%) and Validation (20%)...")
    train_split, val_split = train_test_split(
        train_df, test_size=0.2, random_state=42, stratify=train_df[label_col]
    )

    # Save metadata files
    # For NLP tasks with CSV inputs, the 'metadata' often contains the text itself
    # or is the split dataset.
    train_meta_path = os.path.join(METADATA_DIR, "train.csv")
    val_meta_path = os.path.join(METADATA_DIR, "validation.csv")
    test_meta_path = os.path.join(METADATA_DIR, "test.csv")

    train_split.to_csv(train_meta_path, index=False)
    val_split.to_csv(val_meta_path, index=False)
    test_df.to_csv(test_meta_path, index=False)

    print("Metadata generation complete.")

    # ==========================================
    # Verification Step
    # ==========================================
    print("\nVerifying generated metadata...")

    # Reload data
    df_train_new = pd.read_csv(train_meta_path)
    df_val_new = pd.read_csv(val_meta_path)
    df_test_new = pd.read_csv(test_meta_path)

    # 1. Summary Statistics
    print("-" * 30)
    print("Summary Statistics:")
    print("-" * 30)

    datasets = {"Train": df_train_new, "Validation": df_val_new, "Test": df_test_new}

    for name, df in datasets.items():
        print(f"Dataset: {name}")
        print(f"  Shape: {df.shape}")
        if label_col in df.columns:
            class_dist = df[label_col].value_counts(normalize=True).to_dict()
            print(f"  Class Distribution: {class_dist}")
            print(f"  Positive Samples: {df[label_col].sum()}")
        else:
            print("  Labels: Not present")
        print("-" * 30)

    # 2. File Path Check
    # The dataset consists of text inside CSVs, not external files.
    # Therefore, there are no relative file paths to check against the filesystem.
    # We skip the "missing file ratio" check as it applies to image/audio datasets.

    # 3. Validation Split Verification
    print("Verifying validation split requirements...")

    # Check split ratio
    n_train = len(df_train_new)
    n_val = len(df_val_new)
    total_train_val = n_train + n_val

    actual_val_ratio = n_val / total_train_val
    expected_ratio = 0.20

    print(f"Total Train+Val samples: {total_train_val}")
    print(f"Actual Validation Ratio: {actual_val_ratio:.4f}")

    # Allow a small margin of error for discrete splitting
    if abs(actual_val_ratio - expected_ratio) > 0.01:
        raise AssertionError(
            f"Validation split ratio mismatch. Expected ~0.2, got {actual_val_ratio:.4f}"
        )

    # Check Stratification
    train_mean = df_train_new[label_col].mean()
    val_mean = df_val_new[label_col].mean()

    print(f"Train Label Mean: {train_mean:.4f}")
    print(f"Val Label Mean:   {val_mean:.4f}")

    # Stratification should keep means very close
    if abs(train_mean - val_mean) > 0.01:
        raise AssertionError(
            f"Stratification failed. Train mean: {train_mean:.4f}, Val mean: {val_mean:.4f}"
        )

    print("\nAll checks passed successfully.")


if __name__ == "__main__":
    main()
