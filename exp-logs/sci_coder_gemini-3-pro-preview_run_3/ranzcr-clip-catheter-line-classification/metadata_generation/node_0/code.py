import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

# -----------------------------------------------------------------------------
# Constants & Configuration
# -----------------------------------------------------------------------------
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
TRAIN_CSV_PATH = os.path.join(INPUT_DIR, "train.csv")
SAMPLE_SUB_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")
TRAIN_IMG_DIR = "train"
TEST_IMG_DIR = "test"

RANDOM_STATE = 42
VAL_SIZE = 0.2
CHECK_SAMPLE_SIZE = 1000
MISSING_FILE_THRESHOLD = 0.5


def main():
    # Ensure metadata directory exists
    os.makedirs(METADATA_DIR, exist_ok=True)

    print("Starting metadata generation...")

    # -------------------------------------------------------------------------
    # 1. Load Data
    # -------------------------------------------------------------------------
    if not os.path.exists(TRAIN_CSV_PATH):
        raise FileNotFoundError(f"{TRAIN_CSV_PATH} not found.")

    if not os.path.exists(SAMPLE_SUB_PATH):
        raise FileNotFoundError(f"{SAMPLE_SUB_PATH} not found.")

    df_train_orig = pd.read_csv(TRAIN_CSV_PATH)
    df_test_orig = pd.read_csv(SAMPLE_SUB_PATH)

    print(f"Loaded train.csv: {df_train_orig.shape}")
    print(f"Loaded sample_submission.csv: {df_test_orig.shape}")

    # -------------------------------------------------------------------------
    # 2. Process Training & Validation Data
    # -------------------------------------------------------------------------
    # Construct file paths relative to ./input
    # Format: train/<StudyInstanceUID>.jpg
    df_train_orig["file_path"] = df_train_orig["StudyInstanceUID"].apply(
        lambda x: os.path.join(TRAIN_IMG_DIR, f"{x}.jpg")
    )

    # Perform Group Shuffle Split
    # We must group by PatientID to ensure no patient appears in both train and val
    if "PatientID" not in df_train_orig.columns:
        raise ValueError("PatientID column missing in train.csv")

    splitter = GroupShuffleSplit(
        n_splits=1, test_size=VAL_SIZE, random_state=RANDOM_STATE
    )
    train_idx, val_idx = next(
        splitter.split(df_train_orig, groups=df_train_orig["PatientID"])
    )

    df_train = df_train_orig.iloc[train_idx].copy()
    df_val = df_train_orig.iloc[val_idx].copy()

    # Save to metadata
    train_meta_path = os.path.join(METADATA_DIR, "train.csv")
    val_meta_path = os.path.join(METADATA_DIR, "val.csv")

    df_train.to_csv(train_meta_path, index=False)
    df_val.to_csv(val_meta_path, index=False)

    print(f"Generated {train_meta_path} with {len(df_train)} samples.")
    print(f"Generated {val_meta_path} with {len(df_val)} samples.")

    # -------------------------------------------------------------------------
    # 3. Process Test Data
    # -------------------------------------------------------------------------
    # Construct file paths relative to ./input
    # Format: test/<StudyInstanceUID>.jpg
    df_test_orig["file_path"] = df_test_orig["StudyInstanceUID"].apply(
        lambda x: os.path.join(TEST_IMG_DIR, f"{x}.jpg")
    )

    test_meta_path = os.path.join(METADATA_DIR, "test.csv")
    df_test_orig.to_csv(test_meta_path, index=False)

    print(f"Generated {test_meta_path} with {len(df_test_orig)} samples.")

    # -------------------------------------------------------------------------
    # 4. Verification & Statistics
    # -------------------------------------------------------------------------
    print("\n" + "=" * 40)
    print("VERIFICATION & STATISTICS")
    print("=" * 40)

    # Reload data to verify integrity
    df_train_check = pd.read_csv(train_meta_path)
    df_val_check = pd.read_csv(val_meta_path)
    df_test_check = pd.read_csv(test_meta_path)

    # 4.1 Summary Statistics
    print("\n[Summary Statistics]")
    print(f"Train Set: {len(df_train_check)} samples")
    print(f"Val Set:   {len(df_val_check)} samples")
    print(f"Test Set:  {len(df_test_check)} samples")

    # Identify label columns (exclude IDs and paths)
    non_label_cols = ["StudyInstanceUID", "PatientID", "file_path"]
    label_cols = [c for c in df_train_check.columns if c not in non_label_cols]

    print(f"\nLabel Columns ({len(label_cols)}): {label_cols}")

    print("\nTrain Label Distribution (Mean):")
    print(df_train_check[label_cols].mean())

    print("\nVal Label Distribution (Mean):")
    print(df_val_check[label_cols].mean())

    # 4.2 Verify Group Split (Patient Leakage)
    train_patients = set(df_train_check["PatientID"].unique())
    val_patients = set(df_val_check["PatientID"].unique())
    overlap = train_patients.intersection(val_patients)

    print(f"\n[Split Verification]")
    print(f"Unique Patients in Train: {len(train_patients)}")
    print(f"Unique Patients in Val:   {len(val_patients)}")
    print(f"Patient Overlap Count:    {len(overlap)}")

    if len(overlap) > 0:
        raise AssertionError(
            f"Data leakage detected! {len(overlap)} patients found in both train and validation sets."
        )
    else:
        print("SUCCESS: No patient leakage detected.")

    # 4.3 Verify File Existence
    def verify_files(df, dataset_name):
        print(f"\n[File Existence Check: {dataset_name}]")
        sample_df = df.sample(
            n=min(CHECK_SAMPLE_SIZE, len(df)), random_state=RANDOM_STATE
        )

        missing_files = []
        for _, row in sample_df.iterrows():
            # Path in metadata is relative to ./input
            rel_path = row["file_path"]
            abs_path = os.path.join(INPUT_DIR, rel_path)

            if not os.path.exists(abs_path):
                missing_files.append(rel_path)

        missing_ratio = len(missing_files) / len(sample_df)
        print(
            f"Checked {len(sample_df)} files. Missing: {len(missing_files)} (Ratio: {missing_ratio:.4f})"
        )

        if missing_ratio > MISSING_FILE_THRESHOLD:
            print("Sample of missing files:")
            for p in missing_files[:5]:
                print(f"  - {p}")
            raise FileNotFoundError(
                f"More than {MISSING_FILE_THRESHOLD*100}% of files are missing in {dataset_name} dataset."
            )
        elif missing_ratio > 0:
            print("Warning: Some files are missing, but below threshold.")
        else:
            print("SUCCESS: All checked files exist.")

    verify_files(df_train_check, "Train")
    verify_files(df_val_check, "Validation")
    verify_files(df_test_check, "Test")

    print("\nMetadata generation and verification complete.")


if __name__ == "__main__":
    main()
