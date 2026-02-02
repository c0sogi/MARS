import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Constants
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
TRAIN_DIR = os.path.join(INPUT_DIR, "train2")
TEST_DIR = os.path.join(INPUT_DIR, "test2")
RANDOM_STATE = 42
VAL_SIZE = 0.2


def generate_metadata():
    os.makedirs(METADATA_DIR, exist_ok=True)

    # --- Process Training Data ---
    print("Scanning training files...")
    train_files = glob.glob(os.path.join(TRAIN_DIR, "*.aif"))

    data = []
    for filepath in train_files:
        filename = os.path.basename(filepath)
        # Determine label from filename suffix
        if filename.endswith("_1.aif"):
            label = 1
        elif filename.endswith("_0.aif"):
            label = 0
        else:
            print(f"Warning: Could not determine label for {filename}, skipping.")
            continue

        # Store path relative to ./input
        rel_path = os.path.join("train2", filename)
        data.append({"file_path": rel_path, "label": label})

    df_full_train = pd.DataFrame(data)
    print(f"Found {len(df_full_train)} training samples.")

    # Split into Train and Validation
    print("Splitting data...")
    train_df, val_df = train_test_split(
        df_full_train,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=df_full_train["label"],
    )

    # Save Train and Val Metadata
    train_csv_path = os.path.join(METADATA_DIR, "train.csv")
    val_csv_path = os.path.join(METADATA_DIR, "val.csv")

    train_df.to_csv(train_csv_path, index=False)
    val_df.to_csv(val_csv_path, index=False)
    print(f"Saved train metadata to {train_csv_path}")
    print(f"Saved val metadata to {val_csv_path}")

    # --- Process Test Data ---
    print("Scanning test files...")
    test_files = glob.glob(os.path.join(TEST_DIR, "*.aif"))

    test_data = []
    for filepath in test_files:
        filename = os.path.basename(filepath)
        rel_path = os.path.join("test2", filename)
        # Test data doesn't have labels, but we need the clip name for submission
        test_data.append({"file_path": rel_path, "clip": filename})

    df_test = pd.DataFrame(test_data)
    print(f"Found {len(df_test)} test samples.")

    test_csv_path = os.path.join(METADATA_DIR, "test.csv")
    df_test.to_csv(test_csv_path, index=False)
    print(f"Saved test metadata to {test_csv_path}")


def check_file_paths(df, name):
    print(f"Checking file paths for {name} dataset...")
    # Sample up to 1000 paths
    sample_size = min(1000, len(df))
    sample_paths = (
        df["file_path"].sample(n=sample_size, random_state=RANDOM_STATE).tolist()
    )

    missing_count = 0
    missing_samples = []

    for rel_path in sample_paths:
        full_path = os.path.join(INPUT_DIR, rel_path)
        if not os.path.exists(full_path):
            missing_count += 1
            if len(missing_samples) < 5:
                missing_samples.append(rel_path)

    missing_ratio = missing_count / sample_size
    print(f"Missing file ratio for {name}: {missing_ratio:.4f}")

    if missing_ratio > 0.5:
        print("Sample missing files:")
        for mp in missing_samples:
            print(mp)
        raise FileNotFoundError(
            f"More than 50% of file paths in {name} metadata do not resolve."
        )


def validate_metadata():
    print("\n--- Validating Generated Metadata ---")

    # Load datasets
    train_df = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))
    test_df = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))

    # 1. Print Summary Statistics
    print("\nSummary Statistics:")
    print(f"Train set shape: {train_df.shape}")
    print(
        f"Train label distribution:\n{train_df['label'].value_counts(normalize=True)}"
    )
    print("-" * 20)
    print(f"Val set shape: {val_df.shape}")
    print(f"Val label distribution:\n{val_df['label'].value_counts(normalize=True)}")
    print("-" * 20)
    print(f"Test set shape: {test_df.shape}")

    # 2. Check File Paths
    check_file_paths(train_df, "train")
    check_file_paths(val_df, "val")
    check_file_paths(test_df, "test")

    # 3. Verify Validation Split Requirements
    print("\nVerifying Split Requirements...")

    # Check Split Ratio
    total_train_val = len(train_df) + len(val_df)
    val_ratio = len(val_df) / total_train_val
    print(f"Actual Validation Ratio: {val_ratio:.4f}")

    # Allow small tolerance for integer division rounding
    if not (0.19 < val_ratio < 0.21):
        raise AssertionError(
            f"Validation split ratio {val_ratio:.4f} is not approximately 0.2"
        )

    # Check Stratification
    train_pos_ratio = train_df["label"].mean()
    val_pos_ratio = val_df["label"].mean()

    print(f"Train Positive Class Ratio: {train_pos_ratio:.4f}")
    print(f"Val Positive Class Ratio: {val_pos_ratio:.4f}")

    # Tolerance for stratification difference
    if abs(train_pos_ratio - val_pos_ratio) > 0.01:
        raise AssertionError(
            "Stratification failed: Class distributions in Train and Val differ significantly."
        )

    print("\nAll validation checks passed successfully.")


if __name__ == "__main__":
    generate_metadata()
    validate_metadata()
