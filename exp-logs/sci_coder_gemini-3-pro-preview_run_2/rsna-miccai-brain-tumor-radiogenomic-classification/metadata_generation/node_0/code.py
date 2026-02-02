import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import random

# Configuration
INPUT_DIR = "./input"
TRAIN_DIR = os.path.join(INPUT_DIR, "train")
TEST_DIR = os.path.join(INPUT_DIR, "test")
LABELS_FILE = os.path.join(INPUT_DIR, "train_labels.csv")
METADATA_DIR = "./metadata"
EXCLUDE_IDS = [109, 123, 709]
RANDOM_STATE = 42
VAL_SIZE = 0.2


def generate_metadata():
    os.makedirs(METADATA_DIR, exist_ok=True)

    # ---------------------------------------------------------
    # 1. Process Training Data
    # ---------------------------------------------------------
    print("Processing training data...")
    # Load labels
    df_train_labels = pd.read_csv(LABELS_FILE)

    # Filter exclusions
    print(f"Original training samples: {len(df_train_labels)}")
    df_train_labels = df_train_labels[
        ~df_train_labels["BraTS21ID"].isin(EXCLUDE_IDS)
    ].copy()
    print(f"After exclusions: {len(df_train_labels)}")

    # Format IDs as 5-digit strings
    df_train_labels["BraTS21ID_str"] = df_train_labels["BraTS21ID"].apply(
        lambda x: f"{x:05d}"
    )

    # Verify existence of folders and construct paths
    valid_entries = []
    for idx, row in df_train_labels.iterrows():
        subject_id = row["BraTS21ID_str"]
        subject_dir = os.path.join("train", subject_id)
        full_subject_dir = os.path.join(INPUT_DIR, subject_dir)

        if os.path.isdir(full_subject_dir):
            entry = {
                "BraTS21ID": row["BraTS21ID"],
                "BraTS21ID_str": subject_id,
                "MGMT_value": row["MGMT_value"],
                "path_FLAIR": os.path.join(subject_dir, "FLAIR"),
                "path_T1w": os.path.join(subject_dir, "T1w"),
                "path_T1wCE": os.path.join(subject_dir, "T1wCE"),
                "path_T2w": os.path.join(subject_dir, "T2w"),
            }
            valid_entries.append(entry)

    df_train_full = pd.DataFrame(valid_entries)
    print(f"Valid training samples found on disk: {len(df_train_full)}")

    # Stratified Split
    train_df, val_df = train_test_split(
        df_train_full,
        test_size=VAL_SIZE,
        stratify=df_train_full["MGMT_value"],
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    # ---------------------------------------------------------
    # 2. Process Test Data
    # ---------------------------------------------------------
    print("Processing test data...")
    test_entries = []
    if os.path.exists(TEST_DIR):
        test_ids = sorted(os.listdir(TEST_DIR))
        for subject_id in test_ids:
            # Check if it looks like a subject ID (digits)
            if not subject_id.isdigit():
                continue

            subject_dir = os.path.join("test", subject_id)
            # Try to convert to int for the ID column, handle potential errors
            try:
                id_int = int(subject_id)
            except ValueError:
                continue

            entry = {
                "BraTS21ID": id_int,
                "BraTS21ID_str": subject_id,
                # No MGMT_value for test
                "path_FLAIR": os.path.join(subject_dir, "FLAIR"),
                "path_T1w": os.path.join(subject_dir, "T1w"),
                "path_T1wCE": os.path.join(subject_dir, "T1wCE"),
                "path_T2w": os.path.join(subject_dir, "T2w"),
            }
            test_entries.append(entry)

    test_df = pd.DataFrame(test_entries)
    print(f"Test samples found: {len(test_df)}")

    # ---------------------------------------------------------
    # 3. Save Metadata
    # ---------------------------------------------------------
    train_save_path = os.path.join(METADATA_DIR, "train.csv")
    val_save_path = os.path.join(METADATA_DIR, "val.csv")
    test_save_path = os.path.join(METADATA_DIR, "test.csv")

    train_df.to_csv(train_save_path, index=False)
    val_df.to_csv(val_save_path, index=False)
    test_df.to_csv(test_save_path, index=False)

    print("Metadata files saved.")
    return train_df, val_df, test_df


def verify_metadata(train_df, val_df, test_df):
    print("\n--- Verifying Metadata ---")

    # 1. Summary Statistics
    print(f"Train set shape: {train_df.shape}")
    print(f"Val set shape: {val_df.shape}")
    print(f"Test set shape: {test_df.shape}")

    print("\nClass Distribution (Train):")
    print(train_df["MGMT_value"].value_counts(normalize=True))
    print("\nClass Distribution (Val):")
    print(val_df["MGMT_value"].value_counts(normalize=True))

    # 2. Path Verification
    print("\nChecking file paths...")
    # Collect all paths from all dataframes
    path_cols = ["path_FLAIR", "path_T1w", "path_T1wCE", "path_T2w"]
    all_paths = []

    for df in [train_df, val_df, test_df]:
        if df is not None and not df.empty:
            for col in path_cols:
                if col in df.columns:
                    all_paths.extend(df[col].tolist())

    # Sample 1000 paths (or all if less than 1000)
    sample_size = min(1000, len(all_paths))
    sampled_paths = random.sample(all_paths, sample_size)

    missing_count = 0
    missing_samples = []

    for p in sampled_paths:
        full_path = os.path.join(INPUT_DIR, p)
        if not os.path.exists(full_path):
            missing_count += 1
            if len(missing_samples) < 5:
                missing_samples.append(p)

    missing_ratio = missing_count / sample_size if sample_size > 0 else 0
    print(f"Checked {sample_size} paths. Missing ratio: {missing_ratio:.4f}")

    if missing_ratio > 0.5:
        print("Sample missing paths:", missing_samples)
        raise FileNotFoundError(
            f"Too many file paths are invalid. Missing ratio: {missing_ratio}"
        )

    # 3. Validation Logic Verification
    print("\nVerifying split logic...")
    # Check overlap
    train_ids = set(train_df["BraTS21ID"])
    val_ids = set(val_df["BraTS21ID"])
    overlap = train_ids.intersection(val_ids)
    if overlap:
        raise AssertionError(
            f"Data leakage detected! IDs in both train and val: {overlap}"
        )

    # Check stratification roughly
    train_mean = train_df["MGMT_value"].mean()
    val_mean = val_df["MGMT_value"].mean()
    diff = abs(train_mean - val_mean)
    print(
        f"Label mean - Train: {train_mean:.4f}, Val: {val_mean:.4f}, Diff: {diff:.4f}"
    )

    # Allow a small margin of error for stratification differences due to small dataset size
    if diff > 0.1:
        raise AssertionError(
            "Stratification failed: significant difference in class distribution."
        )

    print("\nAll checks passed successfully.")


if __name__ == "__main__":
    try:
        t_df, v_df, te_df = generate_metadata()

        # Reload to ensure we test exactly what was saved
        t_df_loaded = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
        v_df_loaded = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))
        te_df_loaded = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))

        verify_metadata(t_df_loaded, v_df_loaded, te_df_loaded)

    except Exception as e:
        print(f"Error during execution: {e}")
        raise e
