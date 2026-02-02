import pandas as pd
import numpy as np
import os
from sklearn.model_selection import GroupShuffleSplit

# Constants
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
RANDOM_STATE = 42
VAL_SIZE = 0.2


def generate_metadata():
    # Ensure metadata directory exists
    os.makedirs(METADATA_DIR, exist_ok=True)

    # 1. Load Raw Data
    train_csv_path = os.path.join(INPUT_DIR, "train.csv")
    test_csv_path = os.path.join(INPUT_DIR, "test.csv")
    sub_csv_path = os.path.join(INPUT_DIR, "sample_submission.csv")

    df_train_orig = pd.read_csv(train_csv_path)
    df_test_base = pd.read_csv(test_csv_path)
    df_sub = pd.read_csv(sub_csv_path)

    # 2. Process Training Data
    # Add relative path to DICOM directory (folder level)
    df_train_orig["dcm_path"] = "train/" + df_train_orig["Patient"]

    # Split into Train and Validation using GroupShuffleSplit to keep patients distinct
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=VAL_SIZE, random_state=RANDOM_STATE
    )
    # The split method returns indices
    train_idx, val_idx = next(
        splitter.split(df_train_orig, groups=df_train_orig["Patient"])
    )

    df_train = df_train_orig.iloc[train_idx].copy()
    df_val = df_train_orig.iloc[val_idx].copy()

    # Save Train/Val Metadata
    train_meta_path = os.path.join(METADATA_DIR, "train_metadata.csv")
    val_meta_path = os.path.join(METADATA_DIR, "val_metadata.csv")

    df_train.to_csv(train_meta_path, index=False)
    df_val.to_csv(val_meta_path, index=False)

    # 3. Process Test Data
    # The submission file defines the target (Patient, Week) pairs.
    # We merge this with the static baseline info from test.csv.

    # Extract Patient and Week from "Patient_Week" (e.g., ID000..._12)
    split_ids = df_sub["Patient_Week"].str.rsplit("_", n=1, expand=True)
    df_sub["Patient"] = split_ids[0]
    df_sub["Weeks"] = split_ids[1].astype(int)

    # Rename columns in test_base to clearly indicate they are baseline features
    df_test_base = df_test_base.rename(
        columns={
            "Weeks": "Baseline_Weeks",
            "FVC": "Baseline_FVC",
            "Percent": "Baseline_Percent",
        }
    )

    # Merge submission targets with baseline features
    df_test_meta = pd.merge(df_sub, df_test_base, on="Patient", how="left")

    # Add dcm_path
    df_test_meta["dcm_path"] = "test/" + df_test_meta["Patient"]

    # Save Test Metadata
    test_meta_path = os.path.join(METADATA_DIR, "test_metadata.csv")
    df_test_meta.to_csv(test_meta_path, index=False)

    return train_meta_path, val_meta_path, test_meta_path


def verify_metadata(train_path, val_path, test_path):
    print("Loading generated metadata for verification...")
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    # --- 1. Summary Statistics ---
    print("\n=== Summary Statistics ===")
    datasets = [("Train", df_train), ("Validation", df_val), ("Test", df_test)]
    for name, df in datasets:
        print(f"\n[{name} Set]")
        print(f"Total Samples: {len(df)}")
        print(f"Unique Patients: {df['Patient'].nunique()}")
        print(f"Columns: {list(df.columns)}")
        if (
            "FVC" in df.columns and name != "Test"
        ):  # Test FVC is dummy/placeholder in sub file
            print(f"FVC Mean: {df['FVC'].mean():.2f} | Std: {df['FVC'].std():.2f}")

    # --- 2. Verify Split Strategy ---
    print("\n=== Verifying Split Strategy ===")
    train_patients = set(df_train["Patient"].unique())
    val_patients = set(df_val["Patient"].unique())

    # Check for overlap
    overlap = train_patients.intersection(val_patients)
    if overlap:
        raise AssertionError(
            f"Data Leakage Detected! Patients in both train and val: {overlap}"
        )
    else:
        print("SUCCESS: No patient overlap between Train and Validation sets.")

    # Check split ratio
    total_patients = len(train_patients) + len(val_patients)
    val_ratio = len(val_patients) / total_patients
    print(f"Validation Patient Ratio: {val_ratio:.4f} (Target: {VAL_SIZE})")

    # --- 3. Verify File Paths ---
    print("\n=== Verifying File Paths ===")

    def check_paths(df, name):
        paths = df["dcm_path"].values
        # Sample 1000 paths randomly if dataset is larger
        if len(paths) > 1000:
            paths = np.random.choice(paths, 1000, replace=False)

        missing_count = 0
        missing_samples = []

        for rel_path in paths:
            full_path = os.path.join(INPUT_DIR, rel_path)
            # Check if directory exists (since path points to patient folder)
            if not os.path.exists(full_path):
                missing_count += 1
                if len(missing_samples) < 5:
                    missing_samples.append(rel_path)

        ratio = missing_count / len(paths)
        print(
            f"{name}: Missing Path Ratio = {ratio:.4f} ({missing_count}/{len(paths)})"
        )

        if ratio > 0.5:
            print(f"DEBUG: Sample missing paths: {missing_samples}")
            raise FileNotFoundError(
                f"Critical Error: More than 50% of file paths in {name} are invalid."
            )

    check_paths(df_train, "Train")
    check_paths(df_val, "Validation")
    check_paths(df_test, "Test")

    print("\nVerification Complete: All checks passed.")


if __name__ == "__main__":
    t_path, v_path, ts_path = generate_metadata()
    verify_metadata(t_path, v_path, ts_path)
