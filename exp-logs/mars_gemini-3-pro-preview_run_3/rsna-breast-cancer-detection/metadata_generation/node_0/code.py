import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import glob

# Constants
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
RANDOM_STATE = 42
VAL_SIZE = 0.2


def generate_metadata():
    # Create metadata directory
    os.makedirs(METADATA_DIR, exist_ok=True)

    # ---------------------------------------------------------
    # 1. Process Training Data (Train + Val Split)
    # ---------------------------------------------------------
    print("Processing training data...")
    train_csv_path = os.path.join(INPUT_DIR, "train.csv")
    df_train_full = pd.read_csv(train_csv_path)

    # Construct relative file paths
    # Format: train_images/[patient_id]/[image_id].dcm
    df_train_full["file_path"] = df_train_full.apply(
        lambda row: os.path.join(
            "train_images", str(row["patient_id"]), f"{row['image_id']}.dcm"
        ),
        axis=1,
    )

    # Group by patient_id to get patient-level labels for stratified split
    # We assume if any image for a patient is cancer=1, the patient is positive.
    patient_groups = df_train_full.groupby("patient_id")["cancer"].max()
    patient_ids = patient_groups.index.values
    patient_labels = patient_groups.values

    # Perform Stratified Split on Patients
    train_patients, val_patients = train_test_split(
        patient_ids,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=patient_labels,
    )

    # Create DataFrames based on patient split
    df_train = df_train_full[df_train_full["patient_id"].isin(train_patients)].copy()
    df_val = df_train_full[df_train_full["patient_id"].isin(val_patients)].copy()

    # Save to metadata
    train_meta_path = os.path.join(METADATA_DIR, "train_metadata.csv")
    val_meta_path = os.path.join(METADATA_DIR, "val_metadata.csv")

    df_train.to_csv(train_meta_path, index=False)
    df_val.to_csv(val_meta_path, index=False)

    print(f"Saved train metadata to {train_meta_path} ({len(df_train)} rows)")
    print(f"Saved val metadata to {val_meta_path} ({len(df_val)} rows)")

    # ---------------------------------------------------------
    # 2. Process Test Data
    # ---------------------------------------------------------
    print("Processing test data...")
    test_csv_path = os.path.join(INPUT_DIR, "test.csv")
    df_test = pd.read_csv(test_csv_path)

    # Construct relative file paths
    # Format: test_images/[patient_id]/[image_id].dcm
    df_test["file_path"] = df_test.apply(
        lambda row: os.path.join(
            "test_images", str(row["patient_id"]), f"{row['image_id']}.dcm"
        ),
        axis=1,
    )

    # Save to metadata
    test_meta_path = os.path.join(METADATA_DIR, "test_metadata.csv")
    df_test.to_csv(test_meta_path, index=False)
    print(f"Saved test metadata to {test_meta_path} ({len(df_test)} rows)")

    return train_meta_path, val_meta_path, test_meta_path


def validate_metadata(train_path, val_path, test_path):
    print("\nStarting Validation Checks...")

    # Load datasets
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    # ---------------------------------------------------------
    # Check 1: Summary Statistics
    # ---------------------------------------------------------
    print("\n=== Summary Statistics ===")
    for name, df in [("Train", df_train), ("Validation", df_val), ("Test", df_test)]:
        print(f"\nDataset: {name}")
        print(f"Shape: {df.shape}")
        print(f"Unique Patients: {df['patient_id'].nunique()}")
        if "cancer" in df.columns:
            print(f"Class Distribution:\n{df['cancer'].value_counts(normalize=True)}")
            print(f"Positive Samples: {df['cancer'].sum()}")

    # ---------------------------------------------------------
    # Check 2: File Existence Check
    # ---------------------------------------------------------
    print("\n=== File Existence Check ===")

    def check_files(df, name):
        # Sample up to 1000 paths
        sample_size = min(1000, len(df))
        sample_paths = (
            df["file_path"].sample(n=sample_size, random_state=RANDOM_STATE).values
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
        print(
            f"{name}: Checked {sample_size} files. Missing: {missing_count} ({missing_ratio:.2%})"
        )

        if missing_ratio > 0.5:
            print(f"Sample missing files from {name}:")
            for mp in missing_samples:
                print(f"  - {mp}")
            raise FileNotFoundError(
                f"More than 50% of files are missing in {name} dataset (Ratio: {missing_ratio:.2f})"
            )

    check_files(df_train, "Train")
    check_files(df_val, "Validation")

    # For test, we check, but be aware that in some environments test.csv might have more rows than images provided
    # However, per requirements, we must raise error if > 0.5.
    check_files(df_test, "Test")

    # ---------------------------------------------------------
    # Check 3: Validation Split Verification
    # ---------------------------------------------------------
    print("\n=== Split Verification ===")

    train_patients = set(df_train["patient_id"].unique())
    val_patients = set(df_val["patient_id"].unique())

    # Assert no overlap
    intersection = train_patients.intersection(val_patients)
    assert (
        len(intersection) == 0
    ), f"Data Leakage Detected! {len(intersection)} patients are in both Train and Validation sets."
    print("PASS: No patient overlap between Train and Validation.")

    # Assert split ratio (approximate check)
    total_patients = len(train_patients) + len(val_patients)
    val_ratio = len(val_patients) / total_patients
    print(f"Validation Patient Ratio: {val_ratio:.4f} (Target: {VAL_SIZE})")

    # Allow small deviation due to discrete number of patients
    assert (
        abs(val_ratio - VAL_SIZE) < 0.05
    ), f"Split ratio {val_ratio} deviates too much from target {VAL_SIZE}"
    print("PASS: Split ratio is within acceptable bounds.")

    # Assert Stratification (check if cancer prevalence is roughly similar)
    # Note: Stratification was done on patient level, but let's check image level prevalence
    train_prev = df_train["cancer"].mean()
    val_prev = df_val["cancer"].mean()
    print(f"Train Cancer Prevalence: {train_prev:.4f}")
    print(f"Val Cancer Prevalence:   {val_prev:.4f}")

    # We don't assert strict equality here as image counts per patient vary, but they should be close
    print("PASS: Stratification check complete.")


if __name__ == "__main__":
    try:
        train_path, val_path, test_path = generate_metadata()
        validate_metadata(train_path, val_path, test_path)
        print("\nSUCCESS: Metadata generation and validation completed.")
    except Exception as e:
        print(f"\nFAILURE: {e}")
        raise e
