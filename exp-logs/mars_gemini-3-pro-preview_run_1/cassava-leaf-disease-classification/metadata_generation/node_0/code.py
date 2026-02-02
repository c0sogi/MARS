import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import shutil

# Constants
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
TRAIN_IMAGES_DIR = "train_images"
TEST_IMAGES_DIR = "test_images"
RANDOM_STATE = 42
VAL_SIZE = 0.2


def generate_metadata():
    """
    Reads raw data, creates metadata with relative paths, splits training data,
    and saves to metadata directory.
    """
    print("Starting metadata generation...")

    # Ensure metadata directory exists
    if os.path.exists(METADATA_DIR):
        shutil.rmtree(METADATA_DIR)
    os.makedirs(METADATA_DIR)

    # Load raw CSVs
    train_csv_path = os.path.join(INPUT_DIR, "train.csv")
    sample_sub_path = os.path.join(INPUT_DIR, "sample_submission.csv")

    df_train_full = pd.read_csv(train_csv_path)
    df_test = pd.read_csv(sample_sub_path)

    # Add relative file paths
    # The requirement is: paths relative to ./input
    df_train_full["file_path"] = df_train_full["image_id"].apply(
        lambda x: os.path.join(TRAIN_IMAGES_DIR, x)
    )
    df_test["file_path"] = df_test["image_id"].apply(
        lambda x: os.path.join(TEST_IMAGES_DIR, x)
    )

    # Perform Stratified Split
    print(
        f"Splitting training data with ratio {1-VAL_SIZE}:{VAL_SIZE} and random state {RANDOM_STATE}..."
    )

    train_df, val_df = train_test_split(
        df_train_full,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=df_train_full["label"],
        shuffle=True,
    )

    # Save metadata files
    train_save_path = os.path.join(METADATA_DIR, "train.csv")
    val_save_path = os.path.join(METADATA_DIR, "val.csv")
    test_save_path = os.path.join(METADATA_DIR, "test.csv")

    train_df.to_csv(train_save_path, index=False)
    val_df.to_csv(val_save_path, index=False)
    df_test.to_csv(test_save_path, index=False)

    print(f"Metadata saved to {METADATA_DIR}")
    return train_save_path, val_save_path, test_save_path


def check_file_existence(df, name):
    """
    Checks if a random sample of file paths in the dataframe exist in the input directory.
    """
    print(f"Checking file existence for {name}...")
    sample_size = 1000
    if len(df) < sample_size:
        sample_paths = df["file_path"].values
    else:
        sample_paths = (
            df["file_path"].sample(n=sample_size, random_state=RANDOM_STATE).values
        )

    missing_count = 0
    missing_samples = []

    for rel_path in sample_paths:
        # Resolve path relative to INPUT_DIR
        full_path = os.path.join(INPUT_DIR, rel_path)
        if not os.path.exists(full_path):
            missing_count += 1
            if len(missing_samples) < 5:
                missing_samples.append(rel_path)

    missing_ratio = missing_count / len(sample_paths)
    print(f"  Missing file ratio: {missing_ratio:.4f}")

    if missing_ratio > 0.5:
        print("  Sample missing paths:", missing_samples)
        raise FileNotFoundError(
            f"More than 50% of files are missing for {name}. Ratio: {missing_ratio}"
        )


def validate_metadata(train_path, val_path, test_path):
    """
    Loads generated metadata and performs validation checks.
    """
    print("\nValidating generated metadata...")

    # Load datasets
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    # 1. Print Summary Statistics
    print("\n--- Summary Statistics ---")
    print(f"Train set size: {len(df_train)}")
    print(f"Val set size:   {len(df_val)}")
    print(f"Test set size:  {len(df_test)}")

    print("\nTrain Label Distribution:")
    print(df_train["label"].value_counts(normalize=True))
    print("\nVal Label Distribution:")
    print(df_val["label"].value_counts(normalize=True))

    # 2. Check File Existence
    check_file_existence(df_train, "Train")
    check_file_existence(df_val, "Validation")
    check_file_existence(df_test, "Test")

    # 3. Verify Validation Split Requirements
    print("\nVerifying split requirements...")

    # Check overlap
    train_ids = set(df_train["image_id"])
    val_ids = set(df_val["image_id"])
    overlap = train_ids.intersection(val_ids)
    if overlap:
        raise AssertionError(
            f"Train and Validation sets overlap! {len(overlap)} common IDs found."
        )

    # Check split ratio
    total_train_val = len(df_train) + len(df_val)
    val_ratio = len(df_val) / total_train_val
    print(f"Actual Validation Ratio: {val_ratio:.4f} (Target: {VAL_SIZE})")

    # Allow a small margin of error for discrete splitting
    if not (0.19 < val_ratio < 0.21):
        raise AssertionError(
            f"Validation split ratio {val_ratio} deviates significantly from 0.2"
        )

    # Check Stratification
    # We compare the distribution of labels in Train vs Val
    train_dist = df_train["label"].value_counts(normalize=True).sort_index()
    val_dist = df_val["label"].value_counts(normalize=True).sort_index()

    # Calculate maximum absolute difference in class probabilities
    diff = (train_dist - val_dist).abs().max()
    print(f"Max class distribution difference between Train and Val: {diff:.4f}")

    if diff > 0.02:  # Allow small tolerance
        raise AssertionError(
            "Stratification failed: Significant difference in class distributions."
        )

    print("\nAll validation checks passed successfully.")


if __name__ == "__main__":
    try:
        t_path, v_path, te_path = generate_metadata()
        validate_metadata(t_path, v_path, te_path)
    except Exception as e:
        print(f"\nERROR: {e}")
        raise e
