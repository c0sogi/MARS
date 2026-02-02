import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

# Configuration
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
RANDOM_STATE = 42
VAL_SIZE = 0.2


def main():
    # Ensure metadata directory exists
    os.makedirs(METADATA_DIR, exist_ok=True)

    print("Loading raw datasets...")
    train_csv_path = os.path.join(INPUT_DIR, "train.csv")
    sample_sub_path = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Load data
    # train.csv contains metadata for training images
    df_train_orig = pd.read_csv(train_csv_path)

    # sample_submission.csv contains the IDs for the test set
    df_sample_sub = pd.read_csv(sample_sub_path)

    # Handle column name variations in sample_submission if necessary
    if "image_id" not in df_sample_sub.columns and "ID" in df_sample_sub.columns:
        df_sample_sub.rename(columns={"ID": "image_id"}, inplace=True)

    # --- 1. Prepare Training Data ---
    # Add relative file paths
    df_train_orig["file_path"] = "train/" + df_train_orig["image_id"] + ".dicom"

    # --- 2. Prepare Test Data ---
    # Create test metadata
    df_test = pd.DataFrame()
    df_test["image_id"] = df_sample_sub["image_id"]
    df_test["file_path"] = "test/" + df_test["image_id"] + ".dicom"

    # --- 3. Create Validation Split ---
    # We must split by image_id to avoid data leakage (Group Sampling).
    # We also want to stratify by the class distribution (Stratified Sampling).
    # Strategy: Stratify based on "Has Finding" vs "No Finding" (Class 14).

    unique_img_ids = df_train_orig["image_id"].unique()

    # Determine label for each image for stratification
    # Class 14 is "No finding".
    stratify_labels = []
    for img_id in unique_img_ids:
        # Get all classes associated with this image
        classes = df_train_orig[df_train_orig["image_id"] == img_id]["class_id"].values

        # Label 0: No finding (Class 14 is present)
        # Label 1: Finding (Class 14 is NOT present)
        # Note: Usually "No finding" is exclusive, but we check presence.
        has_finding = 0 if 14 in classes else 1
        stratify_labels.append(has_finding)

    # Perform the split on unique IDs
    train_ids, val_ids = train_test_split(
        unique_img_ids,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify_labels,
    )

    # Filter the original dataframe to create train and val sets
    df_train = df_train_orig[df_train_orig["image_id"].isin(train_ids)].copy()
    df_val = df_train_orig[df_train_orig["image_id"].isin(val_ids)].copy()

    # --- 4. Save Metadata ---
    print("Saving metadata to ./metadata/ ...")
    train_save_path = os.path.join(METADATA_DIR, "train.csv")
    val_save_path = os.path.join(METADATA_DIR, "val.csv")
    test_save_path = os.path.join(METADATA_DIR, "test.csv")

    df_train.to_csv(train_save_path, index=False)
    df_val.to_csv(val_save_path, index=False)
    df_test.to_csv(test_save_path, index=False)

    # --- 5. Verification ---
    print("Verifying generated metadata...")

    # Reload to ensure integrity
    df_train_check = pd.read_csv(train_save_path)
    df_val_check = pd.read_csv(val_save_path)
    df_test_check = pd.read_csv(test_save_path)

    # A. Print Summary Statistics
    print("\nSummary Statistics:")
    print(
        f"Train: {len(df_train_check)} rows, {df_train_check['image_id'].nunique()} unique images"
    )
    print(
        f"Val:   {len(df_val_check)} rows, {df_val_check['image_id'].nunique()} unique images"
    )
    print(
        f"Test:  {len(df_test_check)} rows, {df_test_check['image_id'].nunique()} unique images"
    )

    # B. Verify Split Requirements
    # Check 1: No overlap
    train_img_set = set(df_train_check["image_id"])
    val_img_set = set(df_val_check["image_id"])
    overlap = train_img_set.intersection(val_img_set)
    if len(overlap) > 0:
        raise AssertionError(
            f"Data Leakage detected! {len(overlap)} images found in both Train and Val sets."
        )

    # Check 2: Split Ratio (based on unique images)
    n_train = len(train_img_set)
    n_val = len(val_img_set)
    total = n_train + n_val
    actual_val_ratio = n_val / total
    print(f"Validation Split Ratio (by images): {actual_val_ratio:.4f}")

    if not (0.18 <= actual_val_ratio <= 0.22):
        raise AssertionError(
            f"Validation split ratio {actual_val_ratio:.4f} is outside acceptable range (0.18-0.22)."
        )

    # C. Check File Paths
    def check_files_exist(df, name):
        # Sample up to 1000 paths
        sample_size = min(1000, len(df))
        sample_paths = (
            df["file_path"].sample(n=sample_size, random_state=RANDOM_STATE).values
        )

        missing_count = 0
        missing_examples = []

        for rel_path in sample_paths:
            full_path = os.path.join(INPUT_DIR, rel_path)
            if not os.path.exists(full_path):
                missing_count += 1
                if len(missing_examples) < 5:
                    missing_examples.append(rel_path)

        missing_ratio = missing_count / sample_size
        print(f"Missing file ratio for {name}: {missing_ratio:.4f}")

        if missing_ratio > 0.5:
            print(f"Examples of missing paths in {name}: {missing_examples}")
            raise FileNotFoundError(
                f"Validation failed: More than 50% of files missing in {name} metadata."
            )

    print("\nChecking file paths...")
    check_files_exist(df_train_check, "Train")
    check_files_exist(df_val_check, "Val")
    check_files_exist(df_test_check, "Test")

    print("\nSuccess! Metadata generation and verification complete.")


if __name__ == "__main__":
    main()
