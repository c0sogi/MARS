import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import glob
import random

# Constants
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
TRAIN_LABELS_PATH = os.path.join(INPUT_DIR, "train_labels.csv")
TRAIN_DIR = os.path.join(INPUT_DIR, "train")
TEST_DIR = os.path.join(INPUT_DIR, "test")
EXCLUDE_CASES = [109, 123, 709]
RANDOM_STATE = 42
VAL_SIZE = 0.2


def get_subject_paths(subject_id, base_dir, split_name):
    """
    Generates relative paths for the 4 MRI modalities for a given subject.
    subject_id: int or string representation of ID
    base_dir: 'train' or 'test'
    """
    # Format subject ID to 5 digits (e.g., 0 -> '00000')
    sid_str = str(subject_id).zfill(5)

    # Paths relative to ./input
    # Structure: input/train/00000/FLAIR
    base_rel = os.path.join(split_name, sid_str)

    return {
        "BraTS21ID": subject_id,
        "subject_path": base_rel,
        "flair_path": os.path.join(base_rel, "FLAIR"),
        "t1w_path": os.path.join(base_rel, "T1w"),
        "t1wce_path": os.path.join(base_rel, "T1wCE"),
        "t2w_path": os.path.join(base_rel, "T2w"),
    }


def check_paths(df, name):
    """
    Checks if paths in the dataframe exist.
    """
    print(f"\nChecking file paths for {name} dataset...")

    # Collect all path columns
    path_cols = [c for c in df.columns if c.endswith("_path")]
    all_paths = []
    for col in path_cols:
        all_paths.extend(df[col].tolist())

    # Sample 1000 paths (or all if less than 1000)
    n_samples = min(1000, len(all_paths))
    if n_samples == 0:
        print("No paths to check.")
        return

    sampled_paths = random.sample(all_paths, n_samples)

    missing_count = 0
    missing_samples = []

    for p in sampled_paths:
        # Resolve relative to INPUT_DIR
        full_path = os.path.join(INPUT_DIR, p)
        if not os.path.exists(full_path):
            missing_count += 1
            if len(missing_samples) < 5:
                missing_samples.append(p)

    missing_ratio = missing_count / n_samples
    print(f"Checked {n_samples} paths. Missing ratio: {missing_ratio:.4f}")

    if missing_count > 0:
        print("Sample missing paths:")
        for mp in missing_samples:
            print(f"  {mp}")

    if missing_ratio > 0.5:
        raise FileNotFoundError(
            f"More than 50% of paths are missing in {name} metadata. Ratio: {missing_ratio}"
        )


def main():
    # 1. Setup
    os.makedirs(METADATA_DIR, exist_ok=True)

    # 2. Load Labels
    print("Loading training labels...")
    df_labels = pd.read_csv(TRAIN_LABELS_PATH)
    print(f"Original labels count: {len(df_labels)}")

    # 3. Exclude cases
    df_labels = df_labels[~df_labels["BraTS21ID"].isin(EXCLUDE_CASES)].copy()
    print(f"Labels after exclusion: {len(df_labels)}")

    # 4. Verify presence of folders in train directory
    # Get list of subject folders in train/
    train_folders = os.listdir(TRAIN_DIR)
    # Convert to int for comparison
    train_ids_on_disk = set()
    for f in train_folders:
        if f.isdigit():
            train_ids_on_disk.add(int(f))

    # Filter labels to only include IDs that exist on disk
    df_labels = df_labels[df_labels["BraTS21ID"].isin(train_ids_on_disk)].copy()
    print(f"Labels after verifying disk existence: {len(df_labels)}")

    # 5. Generate Path Metadata for Training Data
    train_data_list = []
    for _, row in df_labels.iterrows():
        sid = row["BraTS21ID"]
        label = row["MGMT_value"]
        info = get_subject_paths(sid, TRAIN_DIR, "train")
        info["MGMT_value"] = label
        train_data_list.append(info)

    df_full_train = pd.DataFrame(train_data_list)

    # 6. Split Train/Validation
    # Stratified split
    X = df_full_train.drop(columns=["MGMT_value"])
    y = df_full_train["MGMT_value"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VAL_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    # Recombine
    df_train = X_train.copy()
    df_train["MGMT_value"] = y_train

    df_val = X_val.copy()
    df_val["MGMT_value"] = y_val

    # 7. Process Test Data
    test_folders = os.listdir(TEST_DIR)
    test_data_list = []
    for f in test_folders:
        if f.isdigit():
            sid = int(f)
            info = get_subject_paths(sid, TEST_DIR, "test")
            # Test data has no label, but we can add a placeholder or leave it out.
            # Instructions imply we predict MGMT_value.
            test_data_list.append(info)

    df_test = pd.DataFrame(test_data_list)

    # 8. Save Metadata
    train_save_path = os.path.join(METADATA_DIR, "train_metadata.csv")
    val_save_path = os.path.join(METADATA_DIR, "val_metadata.csv")
    test_save_path = os.path.join(METADATA_DIR, "test_metadata.csv")

    df_train.to_csv(train_save_path, index=False)
    df_val.to_csv(val_save_path, index=False)
    df_test.to_csv(test_save_path, index=False)

    print(f"Saved metadata to {METADATA_DIR}")

    # ==========================================
    # Verification & Checks
    # ==========================================

    print("\n=== Verification ===")

    # Load back data
    df_train_loaded = pd.read_csv(train_save_path)
    df_val_loaded = pd.read_csv(val_save_path)
    df_test_loaded = pd.read_csv(test_save_path)

    # 1. Summary Statistics
    print(f"Train set size: {len(df_train_loaded)}")
    print(f"Val set size: {len(df_val_loaded)}")
    print(f"Test set size: {len(df_test_loaded)}")

    print("\nTrain Class Distribution:")
    print(df_train_loaded["MGMT_value"].value_counts(normalize=True))

    print("\nVal Class Distribution:")
    print(df_val_loaded["MGMT_value"].value_counts(normalize=True))

    # 2. Check Paths
    check_paths(df_train_loaded, "Train")
    check_paths(df_val_loaded, "Validation")
    check_paths(df_test_loaded, "Test")

    # 3. Verify Split Requirements
    # Check Split Ratio
    total_train_val = len(df_train_loaded) + len(df_val_loaded)
    actual_val_ratio = len(df_val_loaded) / total_train_val
    print(f"\nActual Validation Ratio: {actual_val_ratio:.4f} (Target: {VAL_SIZE})")

    # Allow small deviation due to discrete number of samples
    if not (0.15 < actual_val_ratio < 0.25):
        raise AssertionError(
            f"Validation split ratio {actual_val_ratio} is too far from {VAL_SIZE}"
        )

    # Check Stratification
    train_pos_ratio = df_train_loaded["MGMT_value"].mean()
    val_pos_ratio = df_val_loaded["MGMT_value"].mean()

    print(f"Train Positive Class Ratio: {train_pos_ratio:.4f}")
    print(f"Val Positive Class Ratio: {val_pos_ratio:.4f}")

    # Check if stratification worked reasonably well (difference < 5%)
    if abs(train_pos_ratio - val_pos_ratio) > 0.05:
        raise AssertionError(
            "Stratification failed: Class distributions differ significantly between train and val."
        )

    print("\nAll checks passed successfully.")


if __name__ == "__main__":
    main()
