import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import shutil

# Constants
INPUT_DIR = "./input"
TRAIN_DIR = os.path.join(INPUT_DIR, "train2")
TEST_DIR = os.path.join(INPUT_DIR, "test2")
METADATA_DIR = "./metadata"
RANDOM_STATE = 42
VAL_SIZE = 0.2


def generate_metadata():
    # Ensure metadata directory exists
    if os.path.exists(METADATA_DIR):
        shutil.rmtree(METADATA_DIR)
    os.makedirs(METADATA_DIR)

    # --- Process Training Data ---
    print("Scanning training files...")
    train_files = glob.glob(os.path.join(TRAIN_DIR, "*.aif"))

    data = []
    for filepath in train_files:
        filename = os.path.basename(filepath)
        # Determine label based on suffix
        if filename.endswith("_1.aif"):
            label = 1
        elif filename.endswith("_0.aif"):
            label = 0
        else:
            # Skip files that don't match the expected pattern
            continue

        # Store relative path
        rel_path = os.path.join("train2", filename)
        data.append({"clip_name": filename, "file_path": rel_path, "label": label})

    df_full_train = pd.DataFrame(data)
    print(f"Found {len(df_full_train)} training samples.")

    if len(df_full_train) == 0:
        raise ValueError(
            "No training files found matching the pattern *_0.aif or *_1.aif in input/train2"
        )

    # Stratified Split
    print("Splitting data into train and validation sets...")
    train_df, val_df = train_test_split(
        df_full_train,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=df_full_train["label"],
    )

    # Save Training Metadata
    train_csv_path = os.path.join(METADATA_DIR, "train.csv")
    train_df.to_csv(train_csv_path, index=False)

    # Save Validation Metadata
    val_csv_path = os.path.join(METADATA_DIR, "val.csv")
    val_df.to_csv(val_csv_path, index=False)

    # --- Process Test Data ---
    print("Scanning test files...")
    test_files = glob.glob(os.path.join(TEST_DIR, "*.aif"))

    test_data = []
    for filepath in test_files:
        filename = os.path.basename(filepath)
        rel_path = os.path.join("test2", filename)
        test_data.append({"clip_name": filename, "file_path": rel_path})

    df_test = pd.DataFrame(test_data)
    print(f"Found {len(df_test)} test samples.")

    # Save Test Metadata
    test_csv_path = os.path.join(METADATA_DIR, "test.csv")
    df_test.to_csv(test_csv_path, index=False)

    print("Metadata generation complete.")


def check_file_paths(df, name):
    print(f"Checking file paths for {name}...")
    if "file_path" not in df.columns:
        return

    # Select up to 1000 random samples
    n_samples = min(1000, len(df))
    sample_paths = df["file_path"].sample(n=n_samples, random_state=RANDOM_STATE)

    missing_count = 0
    missing_samples = []

    for rel_path in sample_paths:
        full_path = os.path.join(INPUT_DIR, rel_path)
        if not os.path.exists(full_path):
            missing_count += 1
            if len(missing_samples) < 5:
                missing_samples.append(rel_path)

    missing_ratio = missing_count / n_samples
    print(f"Missing file ratio for {name}: {missing_ratio:.4f}")

    if missing_ratio > 0.5:
        print("Sample missing paths:")
        for p in missing_samples:
            print(f" - {p}")
        raise FileNotFoundError(
            f"More than 50% of file paths in {name} metadata do not exist."
        )


def validate_metadata():
    print("\nStarting validation...")

    # Load metadata
    train_df = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))
    test_df = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))

    # 1. Summary Statistics
    print("\n--- Summary Statistics ---")
    print(f"Train set shape: {train_df.shape}")
    print(f"Validation set shape: {val_df.shape}")
    print(f"Test set shape: {test_df.shape}")

    print("\nTrain Label Distribution:")
    print(train_df["label"].value_counts(normalize=True))
    print("\nValidation Label Distribution:")
    print(val_df["label"].value_counts(normalize=True))

    # 2. Check File Paths
    check_file_paths(train_df, "train")
    check_file_paths(val_df, "validation")
    check_file_paths(test_df, "test")

    # 3. Verify Split Requirements
    print("\nVerifying split requirements...")

    # Verify Split Ratio
    total_train_val = len(train_df) + len(val_df)
    actual_val_ratio = len(val_df) / total_train_val
    print(f"Actual validation ratio: {actual_val_ratio:.4f}")

    # Allow small tolerance for rounding
    if not (0.19 <= actual_val_ratio <= 0.21):
        raise AssertionError(
            f"Validation split ratio {actual_val_ratio:.4f} deviates significantly from 0.2"
        )

    # Verify Stratification
    train_pos_ratio = train_df["label"].mean()
    val_pos_ratio = val_df["label"].mean()

    print(f"Train positive class ratio: {train_pos_ratio:.4f}")
    print(f"Validation positive class ratio: {val_pos_ratio:.4f}")

    # Check if distributions are roughly equal (within 2%)
    if abs(train_pos_ratio - val_pos_ratio) > 0.02:
        raise AssertionError(
            "Stratification failed: Class distributions in train and validation sets differ significantly."
        )

    print("\nAll validation checks passed successfully.")


if __name__ == "__main__":
    try:
        generate_metadata()
        validate_metadata()
    except Exception as e:
        print(f"\nERROR: {e}")
        exit(1)
