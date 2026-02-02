import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import random

# Configuration
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
RANDOM_STATE = 42
VAL_SIZE = 0.2


def generate_metadata():
    print("Starting metadata generation...")

    # Create metadata directory
    os.makedirs(METADATA_DIR, exist_ok=True)

    # Load raw csv files
    train_path = os.path.join(INPUT_DIR, "train.csv")
    test_path = os.path.join(INPUT_DIR, "test.csv")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train file not found at {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test file not found at {test_path}")

    df_train_full = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    # ---------------------------------------------------------
    # 1. Generate Relative File Paths
    # ---------------------------------------------------------
    # Format: [train/test]_images/[patient_id]/[image_id].dcm

    df_train_full["file_path"] = (
        "train_images/"
        + df_train_full["patient_id"].astype(str)
        + "/"
        + df_train_full["image_id"].astype(str)
        + ".dcm"
    )

    df_test["file_path"] = (
        "test_images/"
        + df_test["patient_id"].astype(str)
        + "/"
        + df_test["image_id"].astype(str)
        + ".dcm"
    )

    # ---------------------------------------------------------
    # 2. Create Validation Split (Group Stratified)
    # ---------------------------------------------------------
    # We must split by patient_id to avoid leakage.
    # We also want to stratify by cancer status (if a patient has cancer in any image).

    # Group by patient to determine patient-level labels
    patient_groups = df_train_full.groupby("patient_id")["cancer"].max().reset_index()

    patients = patient_groups["patient_id"].values
    patient_labels = patient_groups["cancer"].values

    # Split patients
    train_patients, val_patients = train_test_split(
        patients, test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=patient_labels
    )

    # Filter original dataframe based on split patients
    df_train = df_train_full[df_train_full["patient_id"].isin(train_patients)].copy()
    df_val = df_train_full[df_train_full["patient_id"].isin(val_patients)].copy()

    # ---------------------------------------------------------
    # 3. Save Metadata
    # ---------------------------------------------------------
    out_train_path = os.path.join(METADATA_DIR, "train.csv")
    out_val_path = os.path.join(METADATA_DIR, "val.csv")
    out_test_path = os.path.join(METADATA_DIR, "test.csv")

    df_train.to_csv(out_train_path, index=False)
    df_val.to_csv(out_val_path, index=False)
    df_test.to_csv(out_test_path, index=False)

    print(f"Metadata saved to {METADATA_DIR}")

    return out_train_path, out_val_path, out_test_path


def check_file_existence(df, name):
    """Checks if a sample of files exists in the input directory."""
    sample_size = 1000
    if len(df) < sample_size:
        sample_paths = df["file_path"].values
    else:
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

    missing_ratio = missing_count / len(sample_paths) if len(sample_paths) > 0 else 0

    print(
        f"[{name}] Checked {len(sample_paths)} files. Missing ratio: {missing_ratio:.4f}"
    )

    if missing_ratio > 0.5:
        print(f"Sample missing files: {missing_samples}")
        raise FileNotFoundError(f"More than 50% of files missing for {name} dataset.")


def verify_metadata(train_path, val_path, test_path):
    print("\nVerifying generated metadata...")

    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    # 1. Print Summary Statistics
    print("-" * 30)
    print(
        f"Train Set: {len(df_train)} samples, {df_train['patient_id'].nunique()} patients"
    )
    print(
        f"Train Class Dist (Image-level): \n{df_train['cancer'].value_counts(normalize=True)}"
    )
    print("-" * 30)
    print(
        f"Val Set:   {len(df_val)} samples, {df_val['patient_id'].nunique()} patients"
    )
    print(
        f"Val Class Dist (Image-level): \n{df_val['cancer'].value_counts(normalize=True)}"
    )
    print("-" * 30)
    print(
        f"Test Set:  {len(df_test)} samples, {df_test['patient_id'].nunique()} patients"
    )
    print("-" * 30)

    # 2. Check File Existence
    check_file_existence(df_train, "Train")
    check_file_existence(df_val, "Val")
    check_file_existence(df_test, "Test")

    # 3. Verify Split Requirements
    train_patients = set(df_train["patient_id"].unique())
    val_patients = set(df_val["patient_id"].unique())

    # Assert no leakage
    overlap = train_patients.intersection(val_patients)
    if overlap:
        raise AssertionError(
            f"Data Leakage detected! {len(overlap)} patients found in both Train and Val."
        )

    # Assert Stratification/Ratio
    total_patients = len(train_patients) + len(val_patients)
    val_ratio = len(val_patients) / total_patients
    print(f"Validation Patient Split Ratio: {val_ratio:.4f} (Target: {VAL_SIZE})")

    if not (0.15 < val_ratio < 0.25):
        raise AssertionError(
            f"Validation split ratio {val_ratio:.4f} is too far from target {VAL_SIZE}"
        )

    # Check stratification consistency (Patient Level)
    train_pos_rate = df_train.groupby("patient_id")["cancer"].max().mean()
    val_pos_rate = df_val.groupby("patient_id")["cancer"].max().mean()

    print(f"Train Patient Pos Rate: {train_pos_rate:.4f}")
    print(f"Val Patient Pos Rate:   {val_pos_rate:.4f}")

    # Allow small variance, but ensure it's not drastically different
    if abs(train_pos_rate - val_pos_rate) > 0.05:
        # Note: With very small datasets or rare classes, this might trigger,
        # but with Stratified split it should be close.
        print("Warning: Class distribution between train and val varies by > 5%.")

    print("\nAll verification checks passed.")


if __name__ == "__main__":
    try:
        t_path, v_path, te_path = generate_metadata()
        verify_metadata(t_path, v_path, te_path)
    except Exception as e:
        print(f"\nExecution failed: {e}")
        raise e
