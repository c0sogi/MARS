import os
import glob
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split

# Constants
INPUT_DIR = "./input"
TRAIN_DIR = os.path.join(INPUT_DIR, "train")
TEST_DIR = os.path.join(INPUT_DIR, "test")
LABELS_FILE = os.path.join(INPUT_DIR, "train_labels.csv")
METADATA_DIR = "./metadata"
EXCLUDE_IDS = ["00109", "00123", "00709"]
RANDOM_STATE = 42
VAL_SIZE = 0.2


def generate_metadata():
    # Ensure metadata directory exists
    os.makedirs(METADATA_DIR, exist_ok=True)

    # 1. Load and Clean Labels
    print("Loading labels...")
    labels_df = pd.read_csv(LABELS_FILE)

    # Format BraTS21ID to 5-digit string to match folder names
    labels_df["BraTS21ID"] = labels_df["BraTS21ID"].apply(lambda x: f"{int(x):05d}")

    # Exclude specific cases
    print(f"Original label count: {len(labels_df)}")
    labels_df = labels_df[~labels_df["BraTS21ID"].isin(EXCLUDE_IDS)].copy()
    print(f"Label count after exclusion: {len(labels_df)}")

    # 2. Define File Scanning Function
    def get_patient_filepaths(patient_id, root_dir):
        """
        Scans the directory for a patient and returns lists of files for each modality.
        Paths are relative to INPUT_DIR.
        """
        patient_dir = os.path.join(root_dir, patient_id)

        # Modalities
        modalities = ["FLAIR", "T1w", "T1wCE", "T2w"]
        paths = {m: [] for m in modalities}

        if not os.path.exists(patient_dir):
            return paths

        for mod in modalities:
            mod_dir = os.path.join(patient_dir, mod)
            if os.path.exists(mod_dir):
                # Get all .dcm files
                files = [f for f in os.listdir(mod_dir) if f.endswith(".dcm")]
                # Sort for consistency
                files.sort()
                # Create relative paths: train/00000/FLAIR/Image-1.dcm
                rel_mod_dir = os.path.relpath(mod_dir, INPUT_DIR)
                paths[mod] = [os.path.join(rel_mod_dir, f) for f in files]

        return paths

    # 3. Generate Train/Val Metadata
    print("Scanning training data...")
    train_data = []

    # Filter labels to only those that exist in the directory
    for idx, row in labels_df.iterrows():
        pid = row["BraTS21ID"]
        target = row["MGMT_value"]

        # Check if folder exists
        if os.path.exists(os.path.join(TRAIN_DIR, pid)):
            paths = get_patient_filepaths(pid, TRAIN_DIR)
            entry = {
                "BraTS21ID": pid,
                "MGMT_value": target,
                "flair_paths": paths["FLAIR"],
                "t1w_paths": paths["T1w"],
                "t1wce_paths": paths["T1wCE"],
                "t2w_paths": paths["T2w"],
            }
            train_data.append(entry)
        else:
            print(f"Warning: Subject {pid} in labels but not in train folder.")

    full_train_df = pd.DataFrame(train_data)

    # Split Train/Val
    print("Splitting train/validation...")
    X = full_train_df
    y = full_train_df["MGMT_value"]

    train_df, val_df = train_test_split(
        X, test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=y, shuffle=True
    )

    # 4. Generate Test Metadata
    print("Scanning test data...")
    if os.path.exists(TEST_DIR):
        test_subjects = [
            d for d in os.listdir(TEST_DIR) if os.path.isdir(os.path.join(TEST_DIR, d))
        ]
        test_subjects.sort()
    else:
        test_subjects = []

    test_data = []
    for pid in test_subjects:
        paths = get_patient_filepaths(pid, TEST_DIR)
        entry = {
            "BraTS21ID": pid,
            "flair_paths": paths["FLAIR"],
            "t1w_paths": paths["T1w"],
            "t1wce_paths": paths["T1wCE"],
            "t2w_paths": paths["T2w"],
        }
        test_data.append(entry)

    test_df = pd.DataFrame(test_data)

    # 5. Save Metadata
    print("Saving metadata to Parquet...")
    train_df.to_parquet(os.path.join(METADATA_DIR, "train.parquet"), index=False)
    val_df.to_parquet(os.path.join(METADATA_DIR, "val.parquet"), index=False)
    test_df.to_parquet(os.path.join(METADATA_DIR, "test.parquet"), index=False)

    print("Metadata generation complete.")


def verify_metadata():
    print("\nStarting Verification...")

    # Load datasets
    train_df = pd.read_parquet(os.path.join(METADATA_DIR, "train.parquet"))
    val_df = pd.read_parquet(os.path.join(METADATA_DIR, "val.parquet"))
    test_df = pd.read_parquet(os.path.join(METADATA_DIR, "test.parquet"))

    # 1. Summary Statistics
    print("-" * 30)
    print("Summary Statistics")
    print("-" * 30)
    print(f"Train set: {len(train_df)} samples")
    print(f"Class distribution:\n{train_df['MGMT_value'].value_counts(normalize=True)}")
    print(f"Val set: {len(val_df)} samples")
    print(f"Class distribution:\n{val_df['MGMT_value'].value_counts(normalize=True)}")
    print(f"Test set: {len(test_df)} samples")

    # 2. Check File Paths
    print("\nChecking file path existence...")
    all_paths = []

    # Collect paths from columns
    path_cols = ["flair_paths", "t1w_paths", "t1wce_paths", "t2w_paths"]

    for df in [train_df, val_df, test_df]:
        if len(df) == 0:
            continue
        for col in path_cols:
            # Explode list columns to get individual file paths
            paths = df[col].explode().dropna().tolist()
            all_paths.extend(paths)

    # Sample 1000 paths
    if len(all_paths) > 1000:
        sampled_paths = random.sample(all_paths, 1000)
    else:
        sampled_paths = all_paths

    missing_count = 0
    missing_samples = []

    for p in sampled_paths:
        full_path = os.path.join(INPUT_DIR, p)
        if not os.path.exists(full_path):
            missing_count += 1
            if len(missing_samples) < 5:
                missing_samples.append(p)

    missing_ratio = missing_count / len(sampled_paths) if len(sampled_paths) > 0 else 0
    print(f"Checked {len(sampled_paths)} paths. Missing ratio: {missing_ratio:.4f}")

    if len(missing_samples) > 0:
        print("Sample missing paths:")
        for p in missing_samples:
            print(p)

    if missing_ratio > 0.5:
        raise FileNotFoundError(
            f"More than 50% of file paths are missing! Ratio: {missing_ratio}"
        )

    # 3. Verify Validation Split
    print("\nVerifying validation split requirements...")

    # Check overlap
    train_ids = set(train_df["BraTS21ID"])
    val_ids = set(val_df["BraTS21ID"])
    overlap = train_ids.intersection(val_ids)
    if overlap:
        raise AssertionError(f"Train and Validation sets overlap! IDs: {overlap}")

    # Check Stratification
    if len(train_df) > 0 and len(val_df) > 0:
        train_mean = train_df["MGMT_value"].mean()
        val_mean = val_df["MGMT_value"].mean()
        print(f"Train Positive Rate: {train_mean:.4f}")
        print(f"Val Positive Rate:   {val_mean:.4f}")

        # Check if distribution is reasonably preserved (within 10%)
        if abs(train_mean - val_mean) > 0.1:
            print("Warning: Stratification difference is notable.")

    print("Verification passed successfully.")


if __name__ == "__main__":
    generate_metadata()
    verify_metadata()
