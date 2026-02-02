import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
import random

# Constants
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
RANDOM_STATE = 42
TRAIN_RATIO = 0.8


def generate_metadata():
    """
    Generates metadata CSVs for train, validation, and test sets.
    """
    # Ensure metadata directory exists
    os.makedirs(METADATA_DIR, exist_ok=True)

    # --- Load Raw Data ---
    print("Loading raw data...")
    train_csv_path = os.path.join(INPUT_DIR, "train.csv")
    test_csv_path = os.path.join(INPUT_DIR, "sample_submission.csv")

    if not os.path.exists(train_csv_path):
        raise FileNotFoundError(f"Train CSV not found at {train_csv_path}")
    if not os.path.exists(test_csv_path):
        raise FileNotFoundError(f"Test CSV not found at {test_csv_path}")

    df_train_full = pd.read_csv(train_csv_path)
    df_test = pd.read_csv(test_csv_path)

    # --- Process Training Data ---
    # Construct relative file paths
    # Format: train/<StudyInstanceUID>.jpg
    df_train_full["file_path"] = df_train_full["StudyInstanceUID"].apply(
        lambda x: os.path.join("train", f"{x}.jpg")
    )

    # --- Split Data (Grouped by PatientID) ---
    print("Splitting data into train and validation sets...")

    # Check for PatientID column
    if "PatientID" not in df_train_full.columns:
        raise ValueError(
            "PatientID column missing in train.csv, cannot perform group split."
        )

    splitter = GroupShuffleSplit(
        n_splits=1, train_size=TRAIN_RATIO, random_state=RANDOM_STATE
    )

    # Get indices for split
    train_idx, val_idx = next(
        splitter.split(df_train_full, y=None, groups=df_train_full["PatientID"])
    )

    df_train = df_train_full.iloc[train_idx].copy()
    df_val = df_train_full.iloc[val_idx].copy()

    # --- Process Test Data ---
    # Construct relative file paths
    # Format: test/<StudyInstanceUID>.jpg
    df_test["file_path"] = df_test["StudyInstanceUID"].apply(
        lambda x: os.path.join("test", f"{x}.jpg")
    )

    # We keep the columns from sample_submission.csv which usually contains the IDs and target columns (initialized to 0)
    # The task requires predicting probabilities, so the test metadata just needs IDs and paths mostly.

    # --- Save Metadata ---
    print("Saving metadata to ./metadata/ ...")

    train_meta_path = os.path.join(METADATA_DIR, "train.csv")
    val_meta_path = os.path.join(METADATA_DIR, "val.csv")
    test_meta_path = os.path.join(METADATA_DIR, "test.csv")

    df_train.to_csv(train_meta_path, index=False)
    df_val.to_csv(val_meta_path, index=False)
    df_test.to_csv(test_meta_path, index=False)

    print("Metadata generation complete.")
    return train_meta_path, val_meta_path, test_meta_path


def verify_metadata(train_path, val_path, test_path):
    """
    Verifies the generated metadata files.
    """
    print("\nStarting verification...")

    # Load datasets
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    # --- 1. Summary Statistics ---
    print("\n=== Summary Statistics ===")
    print(f"Train set shape: {df_train.shape}")
    print(f"Val set shape:   {df_val.shape}")
    print(f"Test set shape:  {df_test.shape}")

    print(f"Train unique patients: {df_train['PatientID'].nunique()}")
    print(f"Val unique patients:   {df_val['PatientID'].nunique()}")

    # Calculate label distribution for Train
    target_cols = [
        c
        for c in df_train.columns
        if c not in ["StudyInstanceUID", "PatientID", "file_path"]
    ]
    print("\nTrain Label Distribution (Positive Rates):")
    # Filter only numeric columns that look like targets
    numeric_targets = df_train[target_cols].select_dtypes(include=np.number).columns
    print(df_train[numeric_targets].mean())

    # --- 2. File Path Verification ---
    print("\n=== File Path Verification ===")

    def check_paths(df, name):
        paths = df["file_path"].tolist()
        # Randomly sample up to 1000 paths
        sample_size = min(1000, len(paths))
        sampled_paths = random.sample(paths, sample_size)

        missing_count = 0
        missing_samples = []

        for p in sampled_paths:
            full_path = os.path.join(INPUT_DIR, p)
            if not os.path.exists(full_path):
                missing_count += 1
                if len(missing_samples) < 5:
                    missing_samples.append(p)

        missing_ratio = missing_count / sample_size
        print(
            f"[{name}] Missing file ratio: {missing_ratio:.4f} ({missing_count}/{sample_size})"
        )

        if missing_ratio > 0.5:
            print(f"Sample missing paths in {name}: {missing_samples}")
            raise FileNotFoundError(f"Too many missing files in {name} dataset.")

    check_paths(df_train, "Train")
    check_paths(df_val, "Val")
    check_paths(df_test, "Test")

    # --- 3. Split Verification ---
    print("\n=== Split Verification ===")

    train_patients = set(df_train["PatientID"].unique())
    val_patients = set(df_val["PatientID"].unique())

    intersection = train_patients.intersection(val_patients)
    overlap_count = len(intersection)

    print(f"Patient overlap between Train and Val: {overlap_count}")

    if overlap_count > 0:
        raise AssertionError(
            f"Data leakage detected! {overlap_count} patients are in both train and val sets."
        )

    # Verify split ratio roughly
    total_train_val = len(df_train) + len(df_val)
    actual_train_ratio = len(df_train) / total_train_val
    print(f"Actual Train Ratio: {actual_train_ratio:.4f} (Target: {TRAIN_RATIO})")

    # Allow small deviation due to group sizes
    if not (0.75 < actual_train_ratio < 0.85):
        print(
            "Warning: Split ratio deviates significantly from 0.8 due to group sizes."
        )

    print("\nVerification passed successfully.")


if __name__ == "__main__":
    try:
        t_path, v_path, te_path = generate_metadata()
        verify_metadata(t_path, v_path, te_path)
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        exit(1)
