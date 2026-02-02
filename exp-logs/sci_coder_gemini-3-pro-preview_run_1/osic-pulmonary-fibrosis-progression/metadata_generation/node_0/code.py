import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit


def main():
    # Configuration
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    RANDOM_STATE = 42

    # Create metadata directory
    os.makedirs(METADATA_DIR, exist_ok=True)

    print("Loading raw data...")
    try:
        train_df = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))
        test_df = pd.read_csv(os.path.join(INPUT_DIR, "test.csv"))
        sample_sub_df = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv"))
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return

    # ==========================================
    # 1. Process Training and Validation Data
    # ==========================================
    print("Processing training data...")

    # Add relative path to DICOM directory
    # The images are stored in folders named after the Patient ID
    train_df["dicom_dir"] = train_df["Patient"].apply(
        lambda x: os.path.join("train", x)
    )

    # Split into Train and Validation sets using GroupShuffleSplit
    # This ensures patients are not split across train/val (Group Sampling)
    splitter = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=RANDOM_STATE)

    # We split based on the 'Patient' group
    train_idx, val_idx = next(splitter.split(train_df, groups=train_df["Patient"]))

    train_meta = train_df.iloc[train_idx].copy()
    val_meta = train_df.iloc[val_idx].copy()

    # Save to metadata
    train_meta.to_csv(os.path.join(METADATA_DIR, "train.csv"), index=False)
    val_meta.to_csv(os.path.join(METADATA_DIR, "val.csv"), index=False)

    # ==========================================
    # 2. Process Test Data
    # ==========================================
    print("Processing test data...")

    # The test metadata should align with the submission requirement (Patient_Week)
    # We start with sample_submission to get the target rows
    test_meta = sample_sub_df.copy()

    # Parse Patient and Predict_Week from Patient_Week column (Format: ID_Week)
    # We split on the last underscore to separate ID and Week
    split_data = test_meta["Patient_Week"].str.rsplit("_", n=1, expand=True)
    test_meta["Patient"] = split_data[0]
    test_meta["Predict_Week"] = split_data[1].astype(int)

    # Prepare test.csv for merging
    # Rename columns to distinguish baseline info from prediction targets
    test_baseline = test_df.rename(
        columns={
            "Weeks": "Baseline_Week",
            "FVC": "Baseline_FVC",
            "Percent": "Baseline_Percent",
            "Age": "Baseline_Age",
            "Sex": "Baseline_Sex",
            "SmokingStatus": "Baseline_SmokingStatus",
        }
    )

    # Merge baseline info onto the submission rows
    test_meta = pd.merge(test_meta, test_baseline, on="Patient", how="left")

    # Add relative path to DICOM directory for test patients
    test_meta["dicom_dir"] = test_meta["Patient"].apply(
        lambda x: os.path.join("test", x)
    )

    # Save to metadata
    test_meta.to_csv(os.path.join(METADATA_DIR, "test.csv"), index=False)

    # ==========================================
    # 3. Verification and Statistics
    # ==========================================
    print("\n" + "=" * 30)
    print("DATASET STATISTICS")
    print("=" * 30)

    datasets = {"Train": train_meta, "Validation": val_meta, "Test": test_meta}

    for name, df in datasets.items():
        print(f"\n[{name} Dataset]")
        print(f"Total Samples: {len(df)}")
        print(f"Unique Patients: {df['Patient'].nunique()}")
        print(f"Columns: {list(df.columns)}")

        # Print label stats if available
        if "FVC" in df.columns and name != "Test":
            print(
                f"FVC (Label) - Mean: {df['FVC'].mean():.2f}, Std: {df['FVC'].std():.2f}"
            )
        elif "Baseline_FVC" in df.columns:
            print(
                f"Baseline FVC - Mean: {df['Baseline_FVC'].mean():.2f}, Std: {df['Baseline_FVC'].std():.2f}"
            )

    print("\n" + "=" * 30)
    print("VERIFICATION CHECKS")
    print("=" * 30)

    # 3a. Verify File Paths
    # Check if the directories referenced in 'dicom_dir' actually exist
    def verify_paths(df, dataset_name):
        paths = df["dicom_dir"].unique()
        n_check = min(len(paths), 1000)

        # Randomly select paths to check
        if n_check > 0:
            paths_to_check = np.random.choice(paths, n_check, replace=False)
        else:
            paths_to_check = []

        missing_count = 0
        missing_samples = []

        for p in paths_to_check:
            full_path = os.path.join(INPUT_DIR, p)
            if not os.path.exists(full_path):
                missing_count += 1
                if len(missing_samples) < 5:
                    missing_samples.append(p)

        ratio = missing_count / n_check if n_check > 0 else 0
        print(f"[{dataset_name}] Checked {n_check} paths. Missing Ratio: {ratio:.4f}")

        if missing_samples:
            print(f"  Sample missing paths: {missing_samples}")

        if ratio > 0.5:
            raise FileNotFoundError(
                f"Error: More than 50% of file paths are missing in {dataset_name} dataset."
            )

    verify_paths(train_meta, "Train")
    verify_paths(val_meta, "Validation")
    verify_paths(test_meta, "Test")

    # 3b. Verify Split Requirements
    print("\n[Split Verification]")
    train_patients = set(train_meta["Patient"].unique())
    val_patients = set(val_meta["Patient"].unique())

    # Check for overlap
    overlap = train_patients.intersection(val_patients)
    print(f"Patient Overlap Count: {len(overlap)}")
    if len(overlap) > 0:
        raise AssertionError(
            f"CRITICAL: Found {len(overlap)} patients overlapping between Train and Validation sets."
        )

    # Check split ratio
    total_patients = train_df["Patient"].nunique()
    n_train = len(train_patients)
    n_val = len(val_patients)
    actual_ratio = n_train / total_patients

    print(f"Total Patients: {total_patients}")
    print(f"Train Patients: {n_train} ({actual_ratio:.2%})")
    print(f"Val Patients:   {n_val} ({1-actual_ratio:.2%})")

    # Allow small deviation due to discrete number of patients
    if not (0.75 <= actual_ratio <= 0.85):
        raise AssertionError(
            f"Split ratio {actual_ratio:.2f} is significantly different from expected 0.80."
        )

    print("\nAll checks passed successfully.")


if __name__ == "__main__":
    main()
