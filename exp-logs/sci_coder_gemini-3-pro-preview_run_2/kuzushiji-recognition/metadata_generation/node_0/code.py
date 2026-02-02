import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
import re
import glob

# Constants
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
TRAIN_CSV = os.path.join(INPUT_DIR, "train.csv")
SAMPLE_SUBMISSION_CSV = os.path.join(INPUT_DIR, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(INPUT_DIR, "train_images")
TEST_IMG_DIR = os.path.join(INPUT_DIR, "test_images")
RANDOM_STATE = 42
VAL_SIZE = 0.2


def extract_group_id(image_id):
    """
    Extracts the book ID (group) from the image_id.
    Assumes format like 'bookID_pageID' or 'bookID-pageID'.
    """
    # Split by first underscore or hyphen
    match = re.split(r"[_-]", image_id, maxsplit=1)
    if match:
        return match[0]
    return image_id


def check_file_paths(df, base_dir, sample_size=1000):
    """
    Checks if a random sample of file paths in the dataframe exist.
    """
    if len(df) == 0:
        return

    sample_n = min(len(df), sample_size)
    sample_paths = df["file_path"].sample(n=sample_n, random_state=RANDOM_STATE)

    missing_count = 0
    missing_samples = []

    for rel_path in sample_paths:
        # rel_path is relative to ./input, so we join with current working directory context
        # But the requirement says paths in metadata are relative to ./input.
        # So to check existence, we check os.path.join("./input", rel_path)
        # Note: rel_path already includes "train_images/..." or "test_images/..."
        # but we need to remove the leading "./input/" if we stored it that way,
        # or handle how it's stored.

        # Let's assume we store "train_images/xxx.jpg" in metadata.
        # Then full path is ./input/train_images/xxx.jpg

        full_path = os.path.join(INPUT_DIR, rel_path)

        # In case the stored path starts with ./input, strip it to avoid duplication if joining
        if rel_path.startswith("./input/"):
            full_path = rel_path
        elif rel_path.startswith("input/"):
            full_path = "./" + rel_path
        else:
            full_path = os.path.join(INPUT_DIR, rel_path)

        if not os.path.exists(full_path):
            missing_count += 1
            if len(missing_samples) < 5:
                missing_samples.append(full_path)

    missing_ratio = missing_count / sample_n

    if missing_ratio > 0.5:
        print("Sample of missing files:")
        for p in missing_samples:
            print(p)
        raise FileNotFoundError(
            f"Missing file ratio {missing_ratio:.2f} exceeds threshold of 0.5"
        )

    print(f"File path check passed. Missing ratio: {missing_ratio:.4f}")


def main():
    # 1. Setup
    os.makedirs(METADATA_DIR, exist_ok=True)

    # 2. Load Data
    print("Loading raw data...")
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(SAMPLE_SUBMISSION_CSV)

    # 3. Construct File Paths
    # Training images are in train_images folder
    # We'll store paths relative to ./input, e.g., "train_images/image_id.jpg"
    train_df["file_path"] = train_df["image_id"].apply(
        lambda x: os.path.join("train_images", x + ".jpg")
    )

    # Test images are in test_images folder
    test_df["file_path"] = test_df["image_id"].apply(
        lambda x: os.path.join("test_images", x + ".jpg")
    )

    # 4. Group Identification
    print("Extracting group IDs...")
    train_df["group_id"] = train_df["image_id"].apply(extract_group_id)

    # 5. Validation Split
    print(f"Splitting data with GroupShuffleSplit (Val size: {VAL_SIZE})...")
    gss = GroupShuffleSplit(n_splits=1, test_size=VAL_SIZE, random_state=RANDOM_STATE)

    train_idx, val_idx = next(gss.split(train_df, groups=train_df["group_id"]))

    train_meta = train_df.iloc[train_idx].copy()
    val_meta = train_df.iloc[val_idx].copy()
    test_meta = test_df.copy()  # Test set is separate

    # 6. Save Metadata
    print("Saving metadata...")
    train_meta.to_csv(os.path.join(METADATA_DIR, "train.csv"), index=False)
    val_meta.to_csv(os.path.join(METADATA_DIR, "val.csv"), index=False)
    test_meta.to_csv(os.path.join(METADATA_DIR, "test.csv"), index=False)

    # 7. Verification and Statistics
    print("\n=== Verification & Statistics ===")

    # Reload to verify
    train_meta_loaded = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
    val_meta_loaded = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))
    test_meta_loaded = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))

    # Stats
    datasets = {
        "Train": train_meta_loaded,
        "Validation": val_meta_loaded,
        "Test": test_meta_loaded,
    }

    for name, df in datasets.items():
        print(f"\n{name} Dataset:")
        print(f"  Samples: {len(df)}")
        if "group_id" in df.columns:
            print(f"  Unique Groups: {df['group_id'].nunique()}")
        if "labels" in df.columns:
            # Simple check on labels (count non-null/valid)
            # Some images might have no labels (NaN or empty string if parsed differently, but here CSV read might make empty string NaN)
            # In this dataset, labels are strings.
            labeled_count = df["labels"].notna().sum()
            print(f"  Labeled Samples: {labeled_count}")

    # Check File Paths
    print("\nChecking file paths...")
    check_file_paths(train_meta_loaded, INPUT_DIR)
    check_file_paths(val_meta_loaded, INPUT_DIR)
    check_file_paths(test_meta_loaded, INPUT_DIR)

    # Verify Split Logic
    print("\nVerifying split logic...")
    train_groups = set(train_meta_loaded["group_id"])
    val_groups = set(val_meta_loaded["group_id"])

    # Check for group leakage
    intersection = train_groups.intersection(val_groups)
    if intersection:
        raise AssertionError(
            f"Group leakage detected! Groups in both train and val: {intersection}"
        )
    else:
        print("Success: No group leakage detected.")

    # Check split ratio roughly
    total_train_val = len(train_meta_loaded) + len(val_meta_loaded)
    actual_val_ratio = len(val_meta_loaded) / total_train_val
    print(f"Actual Validation Ratio: {actual_val_ratio:.4f} (Target: {VAL_SIZE})")

    # Assert ratio is within reasonable bounds (e.g. +/- 5% due to group sizes)
    # Since groups can be large, variance might be higher than random split, but 0.2 is target.
    if not (0.15 < actual_val_ratio < 0.25):
        print(
            f"Warning: Validation ratio {actual_val_ratio:.4f} deviates significantly from {VAL_SIZE} due to group sizes."
        )

    print("\nMetadata generation and verification complete.")


if __name__ == "__main__":
    main()
