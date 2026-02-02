import pandas as pd
import numpy as np
import os
import glob
from sklearn.model_selection import GroupShuffleSplit

# Constants
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
RANDOM_STATE = 42
TRAIN_RATIO = 0.8


def get_group_id(image_id):
    """
    Extracts the book/document ID from the image_id to use as a group for splitting.
    Common formats: '200015779_00134_2' -> '200015779', 'umgy007-028' -> 'umgy007'
    """
    # Split by underscore then hyphen to capture the prefix
    return image_id.split("_")[0].split("-")[0]


def check_file_paths(df, base_dir, sample_size=1000):
    """
    Checks if a random sample of file paths in the dataframe exist.
    """
    if "file_path" not in df.columns:
        return

    # Sample paths (or all if less than sample_size)
    paths_to_check = (
        df["file_path"]
        .sample(n=min(len(df), sample_size), random_state=RANDOM_STATE)
        .tolist()
    )

    missing_count = 0
    missing_samples = []

    for rel_path in paths_to_check:
        full_path = os.path.join(base_dir, rel_path)
        if not os.path.exists(full_path):
            missing_count += 1
            if len(missing_samples) < 5:
                missing_samples.append(rel_path)

    missing_ratio = missing_count / len(paths_to_check) if paths_to_check else 0

    if missing_ratio > 0.5:
        print(f"Sample missing paths: {missing_samples}")
        raise FileNotFoundError(
            f"Missing file ratio {missing_ratio:.2f} exceeds threshold 0.5"
        )

    print(
        f"File path check passed. Missing ratio: {missing_ratio:.4f} (Checked {len(paths_to_check)} files)"
    )


def main():
    # 1. Setup Directories
    if not os.path.exists(METADATA_DIR):
        os.makedirs(METADATA_DIR)
        print(f"Created metadata directory: {METADATA_DIR}")

    # 2. Load Raw Data
    print("Loading raw data...")
    train_csv_path = os.path.join(INPUT_DIR, "train.csv")
    sample_sub_path = os.path.join(INPUT_DIR, "sample_submission.csv")

    df_train_full = pd.read_csv(train_csv_path)
    df_test = pd.read_csv(sample_sub_path)

    # Handle missing labels (images with no characters)
    df_train_full["labels"] = df_train_full["labels"].fillna("")

    # 3. Add File Paths
    # Note: Dataset structure indicates images are in train_images/ and test_images/
    # The file extension is .jpg based on the dataset info provided.
    df_train_full["file_path"] = "train_images/" + df_train_full["image_id"] + ".jpg"
    df_test["file_path"] = "test_images/" + df_test["image_id"] + ".jpg"

    # 4. Create Groups for Splitting
    # We group by the book ID to prevent data leakage (pages from same book in both train and val)
    df_train_full["group_id"] = df_train_full["image_id"].apply(get_group_id)

    print(
        f"Identified {df_train_full['group_id'].nunique()} unique groups (books) in training data."
    )

    # 5. Split Training Data (Group Shuffle Split)
    splitter = GroupShuffleSplit(
        n_splits=1, train_size=TRAIN_RATIO, random_state=RANDOM_STATE
    )
    train_idx, val_idx = next(
        splitter.split(df_train_full, groups=df_train_full["group_id"])
    )

    df_train = df_train_full.iloc[train_idx].copy()
    df_val = df_train_full.iloc[val_idx].copy()

    # 6. Save Metadata
    print("Saving metadata files...")
    train_meta_path = os.path.join(METADATA_DIR, "train_metadata.csv")
    val_meta_path = os.path.join(METADATA_DIR, "val_metadata.csv")
    test_meta_path = os.path.join(METADATA_DIR, "test_metadata.csv")

    df_train.to_csv(train_meta_path, index=False)
    df_val.to_csv(val_meta_path, index=False)
    df_test.to_csv(test_meta_path, index=False)

    # 7. Reload and Validate
    print("\n--- Validation & Statistics ---")

    # Reload
    df_train_loaded = pd.read_csv(train_meta_path)
    df_val_loaded = pd.read_csv(val_meta_path)
    df_test_loaded = pd.read_csv(test_meta_path)

    # Summary Statistics
    print(f"Train set shape: {df_train_loaded.shape}")
    print(f"Val set shape:   {df_val_loaded.shape}")
    print(f"Test set shape:  {df_test_loaded.shape}")

    print(f"Train unique groups: {df_train_loaded['group_id'].nunique()}")
    print(f"Val unique groups:   {df_val_loaded['group_id'].nunique()}")

    # Check File Paths
    print("\nChecking file paths...")
    check_file_paths(df_train_loaded, INPUT_DIR)
    check_file_paths(df_val_loaded, INPUT_DIR)
    check_file_paths(df_test_loaded, INPUT_DIR)

    # Verify Split Logic
    print("\nVerifying split logic...")
    train_groups = set(df_train_loaded["group_id"].unique())
    val_groups = set(df_val_loaded["group_id"].unique())

    # Check for intersection
    group_intersection = train_groups.intersection(val_groups)
    if group_intersection:
        raise AssertionError(
            f"Split failed: {len(group_intersection)} groups overlap between train and val. Groups: {list(group_intersection)[:5]}..."
        )
    else:
        print("Success: No group overlap between train and validation sets.")

    # Check split ratio
    total_samples = len(df_train_loaded) + len(df_val_loaded)
    actual_train_ratio = len(df_train_loaded) / total_samples
    print(f"Actual Split Ratio: {actual_train_ratio:.4f} (Target: {TRAIN_RATIO})")

    # Allow small deviation due to group sizes
    if abs(actual_train_ratio - TRAIN_RATIO) > 0.1:
        print(
            "Warning: Split ratio deviates significantly from target due to large group sizes."
        )

    print("\nMetadata generation and validation completed successfully.")


if __name__ == "__main__":
    main()
