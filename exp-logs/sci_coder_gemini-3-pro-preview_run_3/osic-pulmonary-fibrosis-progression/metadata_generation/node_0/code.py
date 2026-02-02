import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

# Configuration
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
RANDOM_STATE = 42
VAL_SIZE = 0.2


def generate_metadata():
    """Generates metadata files for train, val, and test sets."""
    print("Generating metadata...")
    os.makedirs(METADATA_DIR, exist_ok=True)

    # Load raw data
    train_df = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(INPUT_DIR, "test.csv"))

    # Add image paths
    # The dataset structure is input/train/<PatientID> and input/test/<PatientID>
    # We store the relative path to the directory containing the DICOM slices
    train_df["image_path"] = train_df["Patient"].apply(
        lambda x: os.path.join("train", x)
    )
    test_df["image_path"] = test_df["Patient"].apply(lambda x: os.path.join("test", x))

    # Split training data into train and validation sets
    # We must split by Patient ID (Group Sampling) to avoid data leakage
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=VAL_SIZE, random_state=RANDOM_STATE
    )
    groups = train_df["Patient"]

    train_idx, val_idx = next(splitter.split(train_df, groups=groups))

    train_meta = train_df.iloc[train_idx].copy()
    val_meta = train_df.iloc[val_idx].copy()
    test_meta = test_df.copy()

    # Save metadata
    train_meta.to_csv(os.path.join(METADATA_DIR, "train.csv"), index=False)
    val_meta.to_csv(os.path.join(METADATA_DIR, "val.csv"), index=False)
    test_meta.to_csv(os.path.join(METADATA_DIR, "test.csv"), index=False)

    print("Metadata generation complete.")


def check_file_paths(df, dataset_name):
    """Checks if a sample of file paths in the dataframe exist."""
    paths = df["image_path"].values
    n_total = len(paths)
    n_check = min(1000, n_total)

    # Randomly sample paths
    rng = np.random.default_rng(RANDOM_STATE)
    sample_paths = rng.choice(paths, size=n_check, replace=False)

    missing_count = 0
    sample_missing = []

    for rel_path in sample_paths:
        full_path = os.path.join(INPUT_DIR, rel_path)
        # Check if the directory exists
        if not os.path.exists(full_path):
            missing_count += 1
            if len(sample_missing) < 5:
                sample_missing.append(rel_path)

    ratio = missing_count / n_check
    print(
        f"[{dataset_name}] Checked {n_check}/{n_total} paths. Missing ratio: {ratio:.4f}"
    )

    if ratio > 0.5:
        print(f"[{dataset_name}] Example missing paths: {sample_missing}")
        raise FileNotFoundError(
            f"[{dataset_name}] Missing file ratio {ratio:.4f} exceeds threshold 0.5"
        )


def validate_metadata():
    """Performs validation checks on the generated metadata."""
    print("\nValidating metadata...")

    # Load generated metadata
    train_meta = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
    val_meta = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))
    test_meta = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))

    # 1. Summary Statistics
    print("--- Summary Statistics ---")
    print(
        f"Train: {len(train_meta)} rows, {train_meta['Patient'].nunique()} unique patients"
    )
    print(
        f"Val:   {len(val_meta)} rows, {val_meta['Patient'].nunique()} unique patients"
    )
    print(
        f"Test:  {len(test_meta)} rows, {test_meta['Patient'].nunique()} unique patients"
    )

    print(f"Train FVC Mean: {train_meta['FVC'].mean():.2f}")
    print(f"Val FVC Mean:   {val_meta['FVC'].mean():.2f}")
    print(f"Test FVC Mean:  {test_meta['FVC'].mean():.2f}")

    # 2. Verify Split Logic
    print("\n--- Verifying Split ---")
    train_patients = set(train_meta["Patient"].unique())
    val_patients = set(val_meta["Patient"].unique())

    # Check for intersection
    intersection = train_patients.intersection(val_patients)
    if intersection:
        raise AssertionError(
            f"Data leakage detected! Patients found in both train and val: {intersection}"
        )

    # Check split ratio
    total_patients = len(train_patients) + len(val_patients)
    actual_val_ratio = len(val_patients) / total_patients
    print(
        f"Actual Validation Ratio (by patient): {actual_val_ratio:.4f} (Target: {VAL_SIZE})"
    )

    # 3. Check File Paths
    print("\n--- Verifying File Paths ---")
    check_file_paths(train_meta, "Train")
    check_file_paths(val_meta, "Val")
    check_file_paths(test_meta, "Test")

    print("\nAll validation checks passed successfully.")


if __name__ == "__main__":
    generate_metadata()
    validate_metadata()
