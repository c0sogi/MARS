import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit


def main():
    # Configuration
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    RANDOM_STATE = 42
    VAL_SIZE = 0.2

    # Ensure metadata directory exists
    os.makedirs(METADATA_DIR, exist_ok=True)

    print("Loading raw data...")
    train_csv_path = os.path.join(INPUT_DIR, "train.csv")
    sample_submission_path = os.path.join(INPUT_DIR, "sample_submission.csv")

    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(sample_submission_path)

    # Construct file paths relative to INPUT_DIR
    # Training images are in 'train/' folder, Test images in 'test/' folder
    # Filenames are StudyInstanceUID + '.jpg'

    train_df["file_path"] = train_df["StudyInstanceUID"].apply(
        lambda x: os.path.join("train", f"{x}.jpg")
    )

    # For test data, we only really need StudyInstanceUID and file_path for loading
    # We keep the structure of sample_submission but add file_path
    test_df["file_path"] = test_df["StudyInstanceUID"].apply(
        lambda x: os.path.join("test", f"{x}.jpg")
    )

    print("Splitting training data into train/val...")
    # Use GroupShuffleSplit to ensure patients are not split across train/val
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=VAL_SIZE, random_state=RANDOM_STATE
    )

    # We need to map the generator to indices
    train_idx, val_idx = next(splitter.split(train_df, groups=train_df["PatientID"]))

    train_split = train_df.iloc[train_idx].copy()
    val_split = train_df.iloc[val_idx].copy()

    print(f"Train split shape: {train_split.shape}")
    print(f"Val split shape: {val_split.shape}")

    # Save metadata
    print("Saving metadata...")
    train_metadata_path = os.path.join(METADATA_DIR, "train_metadata.csv")
    val_metadata_path = os.path.join(METADATA_DIR, "val_metadata.csv")
    test_metadata_path = os.path.join(METADATA_DIR, "test_metadata.csv")

    train_split.to_csv(train_metadata_path, index=False)
    val_split.to_csv(val_metadata_path, index=False)
    test_df.to_csv(test_metadata_path, index=False)

    # ==========================================
    # Validation and Checks
    # ==========================================
    print("\nPerforming validation checks...")

    # Reload datasets to simulate usage
    df_train_check = pd.read_csv(train_metadata_path)
    df_val_check = pd.read_csv(val_metadata_path)
    df_test_check = pd.read_csv(test_metadata_path)

    # 1. Summary Statistics
    print("\n--- Summary Statistics ---")
    print(
        f"Training Set: {len(df_train_check)} samples, {df_train_check['PatientID'].nunique()} unique patients"
    )
    print(
        f"Validation Set: {len(df_val_check)} samples, {df_val_check['PatientID'].nunique()} unique patients"
    )
    print(f"Test Set: {len(df_test_check)} samples")

    # Label distribution in Train/Val
    # Identify label columns (exclude ID, PatientID, file_path)
    non_label_cols = ["StudyInstanceUID", "PatientID", "file_path"]
    label_cols = [c for c in df_train_check.columns if c not in non_label_cols]

    print("\nLabel Distribution (Train vs Val):")
    for col in label_cols:
        train_mean = df_train_check[col].mean()
        val_mean = df_val_check[col].mean()
        print(f"  {col}: Train={train_mean:.4f}, Val={val_mean:.4f}")

    # 2. Check File Paths
    def check_paths(df, name):
        print(f"\nChecking file paths for {name}...")
        # Sample 1000 paths or all if less than 1000
        n_sample = min(1000, len(df))
        sample_paths = df["file_path"].sample(n=n_sample, random_state=RANDOM_STATE)

        missing_count = 0
        missing_samples = []

        for rel_path in sample_paths:
            full_path = os.path.join(INPUT_DIR, rel_path)
            if not os.path.exists(full_path):
                missing_count += 1
                if len(missing_samples) < 5:
                    missing_samples.append(rel_path)

        missing_ratio = missing_count / n_sample
        print(f"  Missing file ratio: {missing_ratio:.4f} ({missing_count}/{n_sample})")

        if missing_count > 0:
            print(f"  Sample missing paths: {missing_samples}")

        if missing_ratio > 0.5:
            raise FileNotFoundError(
                f"Error: More than 50% of file paths in {name} are missing."
            )

    check_paths(df_train_check, "Train Metadata")
    check_paths(df_val_check, "Val Metadata")
    check_paths(df_test_check, "Test Metadata")

    # 3. Verify Split Requirements
    print("\nVerifying split requirements...")

    # Check for PatientID leakage
    train_patients = set(df_train_check["PatientID"].unique())
    val_patients = set(df_val_check["PatientID"].unique())

    intersection = train_patients.intersection(val_patients)
    if intersection:
        raise AssertionError(
            f"Data Leakage Detected: {len(intersection)} patients found in both train and validation sets."
        )
    else:
        print("  Success: No patient overlap between train and validation sets.")

    # Check split ratio roughly
    total_train_val = len(df_train_check) + len(df_val_check)
    actual_val_ratio = len(df_val_check) / total_train_val
    print(f"  Actual Validation Ratio: {actual_val_ratio:.4f} (Target: {VAL_SIZE})")

    # We don't enforce exact ratio strictness because group split depends on group sizes,
    # but it should be reasonably close.

    print("\nMetadata generation and validation completed successfully.")


if __name__ == "__main__":
    main()
