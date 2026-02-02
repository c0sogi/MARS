import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, GroupShuffleSplit

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
RANDOM_STATE = 42

TRAIN_JSON = os.path.join(INPUT_DIR, "iwildcam2020_train_annotations.json")
TEST_JSON = os.path.join(INPUT_DIR, "iwildcam2020_test_information.json")

# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------


def check_file_existence(df, sample_size=1000):
    """
    Checks if a random sample of files in the DataFrame exists on disk.
    Raises an error if the missing ratio > 0.5.
    """
    if df.empty:
        return

    n = min(sample_size, len(df))
    sample = df.sample(n=n, random_state=RANDOM_STATE)

    missing_paths = []
    for _, row in sample.iterrows():
        # Ensure we check the full path relative to current working directory
        file_path = os.path.join(INPUT_DIR, row["file_path"])
        if not os.path.exists(file_path):
            missing_paths.append(row["file_path"])

    missing_ratio = len(missing_paths) / n
    print(f"  Missing file ratio: {missing_ratio:.4f} ({len(missing_paths)}/{n})")

    if missing_ratio > 0.5:
        print("  Sample of missing files:")
        for p in missing_paths[:5]:
            print(f"    {p}")
        raise FileNotFoundError(
            f"More than 50% of sampled files are missing. Check file paths."
        )


def verify_split(train_df, val_df, split_col, split_method):
    """
    Verifies that the train/val split respects the split method (Group or Stratified).
    """
    print(f"  Verifying {split_method} split...")

    train_ids = set(train_df["id"])
    val_ids = set(val_df["id"])

    # 1. Check for ID overlap (basic leak check)
    overlap = train_ids.intersection(val_ids)
    assert (
        len(overlap) == 0
    ), f"Found {len(overlap)} images appearing in both train and validation sets!"

    # 2. Check specific split requirements
    if split_method == "group":
        # Groups (locations) should be disjoint
        train_groups = set(train_df[split_col])
        val_groups = set(val_df[split_col])
        group_overlap = train_groups.intersection(val_groups)
        assert (
            len(group_overlap) == 0
        ), f"Group leakage detected! {len(group_overlap)} locations are in both sets."
        print("  Group split valid: No location overlap.")

    elif split_method == "stratified":
        # Check if validation set is roughly 20%
        total = len(train_df) + len(val_df)
        val_ratio = len(val_df) / total
        assert (
            0.15 < val_ratio < 0.25
        ), f"Validation ratio {val_ratio:.2f} is far from 0.2"
        print("  Stratified split valid: Ratio is acceptable.")


def print_stats(name, df):
    print(f"\n[{name.upper()} DATASET]")
    print(f"  Shape: {df.shape}")
    print(f"  Unique Images: {df['id'].nunique()}")
    if "category_id" in df.columns:
        print(f"  Unique Categories: {df['category_id'].nunique()}")
        print(f"  Top 5 Categories:\n{df['category_id'].value_counts().head(5)}")
    if "location" in df.columns:
        print(f"  Unique Locations: {df['location'].nunique()}")


# ------------------------------------------------------------------------------
# Main Execution
# ------------------------------------------------------------------------------


def main():
    # Create metadata directory
    if not os.path.exists(METADATA_DIR):
        os.makedirs(METADATA_DIR)
        print(f"Created {METADATA_DIR}")

    # --------------------------------------------------------------------------
    # 1. Load and Process Training Data
    # --------------------------------------------------------------------------
    print(f"Loading {TRAIN_JSON}...")
    with open(TRAIN_JSON, "r") as f:
        train_data = json.load(f)

    # Convert to DataFrames
    df_train_imgs = pd.DataFrame(train_data["images"])
    df_train_anns = pd.DataFrame(train_data["annotations"])

    # Merge images and annotations
    # Note: Using left join to keep all images.
    # If an image has no annotation, category_id will be NaN (handled later).
    # We rename 'id' in annotations to 'ann_id' to avoid conflict with image 'id'.
    if "id" in df_train_anns.columns:
        df_train_anns.rename(columns={"id": "ann_id"}, inplace=True)

    # Merge on image id
    df_train = pd.merge(
        df_train_imgs, df_train_anns, left_on="id", right_on="image_id", how="left"
    )

    # Construct file paths
    # The dataset structure indicates images are in 'train/' folder
    df_train["file_path"] = df_train["file_name"].apply(
        lambda x: os.path.join("train", x)
    )

    # Handle missing categories (if any).
    # The task description says 0 represents absence of animal.
    # If category_id is NaN, it might mean no annotation provided or empty.
    # We'll fill with 0 and ensure integer type.
    df_train["category_id"] = df_train["category_id"].fillna(0).astype(int)

    # --------------------------------------------------------------------------
    # 2. Load and Process Test Data
    # --------------------------------------------------------------------------
    print(f"Loading {TEST_JSON}...")
    with open(TEST_JSON, "r") as f:
        test_data = json.load(f)

    df_test = pd.DataFrame(test_data["images"])

    # Construct file paths for test
    df_test["file_path"] = df_test["file_name"].apply(lambda x: os.path.join("test", x))

    # --------------------------------------------------------------------------
    # 3. Split Training Data (Train/Val)
    # --------------------------------------------------------------------------
    # Determine split strategy
    # If 'location' exists, use GroupShuffleSplit to prevent leakage.
    # Otherwise, use StratifiedShuffleSplit.

    # In iWildCam, 'location' is the standard field for camera location.
    split_col = "location" if "location" in df_train.columns else None

    if split_col:
        print(f"Found '{split_col}' column. Performing Group Split (80/20)...")
        splitter = GroupShuffleSplit(
            n_splits=1, train_size=0.8, random_state=RANDOM_STATE
        )
        groups = df_train[split_col]
        # We need to split based on groups.
        # Note: If one image has multiple annotations (rows), they share the same location,
        # so they will stay together.
        train_idx, val_idx = next(
            splitter.split(df_train, df_train["category_id"], groups=groups)
        )
        split_method = "group"
    else:
        print("Location column not found. Performing Stratified Split (80/20)...")
        splitter = StratifiedShuffleSplit(
            n_splits=1, train_size=0.8, random_state=RANDOM_STATE
        )
        train_idx, val_idx = next(splitter.split(df_train, df_train["category_id"]))
        split_method = "stratified"

    train_set = df_train.iloc[train_idx].copy()
    val_set = df_train.iloc[val_idx].copy()

    # --------------------------------------------------------------------------
    # 4. Save Metadata
    # --------------------------------------------------------------------------
    train_csv = os.path.join(METADATA_DIR, "train.csv")
    val_csv = os.path.join(METADATA_DIR, "val.csv")
    test_csv = os.path.join(METADATA_DIR, "test.csv")

    print(f"Saving metadata to {METADATA_DIR}...")
    train_set.to_csv(train_csv, index=False)
    val_set.to_csv(val_csv, index=False)
    df_test.to_csv(test_csv, index=False)

    # --------------------------------------------------------------------------
    # 5. Verification
    # --------------------------------------------------------------------------
    print("\nRunning Verification Checks...")

    # Reload to ensure we check what was saved
    df_train_check = pd.read_csv(train_csv)
    df_val_check = pd.read_csv(val_csv)
    df_test_check = pd.read_csv(test_csv)

    # Print Stats
    print_stats("Train", df_train_check)
    print_stats("Validation", df_val_check)
    print_stats("Test", df_test_check)

    # Check Files
    print("\nChecking file paths...")
    check_file_existence(df_train_check)
    check_file_existence(df_val_check)
    check_file_existence(df_test_check)

    # Verify Split
    print("\nVerifying split logic...")
    # We pass the original dataframe (reconstructed or just use the split ones to check disjointness)
    # To check disjointness properly, we need the split column.
    verify_split(df_train_check, df_val_check, split_col, split_method)

    print("\nAll checks passed successfully.")


if __name__ == "__main__":
    main()
