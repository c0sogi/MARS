import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Constants
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
RANDOM_STATE = 42
TRAIN_VAL_SPLIT_RATIO = 0.2


def generate_metadata():
    print("Starting metadata generation...")

    # Ensure metadata directory exists
    os.makedirs(METADATA_DIR, exist_ok=True)

    # Load raw data
    train_csv_path = os.path.join(INPUT_DIR, "train.csv")
    sample_sub_path = os.path.join(INPUT_DIR, "sample_submission.csv")

    if not os.path.exists(train_csv_path):
        raise FileNotFoundError(f"Could not find {train_csv_path}")
    if not os.path.exists(sample_sub_path):
        raise FileNotFoundError(f"Could not find {sample_sub_path}")

    df_train_full = pd.read_csv(train_csv_path)
    df_test = pd.read_csv(sample_sub_path)

    # Add relative file paths
    # Train images are in train_images/
    df_train_full["file_path"] = df_train_full["image_id"].apply(
        lambda x: os.path.join("train_images", x)
    )

    # Test images are in test_images/
    df_test["file_path"] = df_test["image_id"].apply(
        lambda x: os.path.join("test_images", x)
    )

    # Perform Stratified Split
    print(
        f"Splitting training data with ratio {TRAIN_VAL_SPLIT_RATIO} and random state {RANDOM_STATE}..."
    )

    train_df, val_df = train_test_split(
        df_train_full,
        test_size=TRAIN_VAL_SPLIT_RATIO,
        stratify=df_train_full["label"],
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    # Save metadata
    train_meta_path = os.path.join(METADATA_DIR, "train.csv")
    val_meta_path = os.path.join(METADATA_DIR, "val.csv")
    test_meta_path = os.path.join(METADATA_DIR, "test.csv")

    train_df.to_csv(train_meta_path, index=False)
    val_df.to_csv(val_meta_path, index=False)
    df_test.to_csv(test_meta_path, index=False)

    print(f"Metadata saved to {METADATA_DIR}")
    return train_meta_path, val_meta_path, test_meta_path


def check_file_existence(df, name):
    print(f"Checking file existence for {name}...")
    # Select up to 1000 random samples
    n_samples = min(len(df), 1000)
    if n_samples > 0:
        sample_paths = (
            df["file_path"].sample(n=n_samples, random_state=RANDOM_STATE).values
        )
    else:
        sample_paths = []

    missing_count = 0
    missing_samples = []

    for rel_path in sample_paths:
        full_path = os.path.join(INPUT_DIR, rel_path)
        if not os.path.exists(full_path):
            missing_count += 1
            if len(missing_samples) < 5:
                missing_samples.append(rel_path)

    missing_ratio = missing_count / n_samples if n_samples > 0 else 0
    print(f"  Missing ratio: {missing_ratio:.4f} ({missing_count}/{n_samples})")

    if missing_ratio > 0.5:
        print("  Sample missing paths:", missing_samples)
        raise FileNotFoundError(f"More than 50% of files missing in {name} dataset.")


def validate_metadata(train_path, val_path, test_path):
    print("\nValidating generated metadata...")

    # Load generated metadata
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    # 1. Summary Statistics
    print("\n--- Summary Statistics ---")
    print(f"Train set shape: {df_train.shape}")
    print(f"Val set shape:   {df_val.shape}")
    print(f"Test set shape:  {df_test.shape}")

    print("\nTrain Class Distribution:")
    print(df_train["label"].value_counts(normalize=True).sort_index())
    print("\nVal Class Distribution:")
    print(df_val["label"].value_counts(normalize=True).sort_index())

    # 2. File Path Checks
    check_file_existence(df_train, "Train")
    check_file_existence(df_val, "Validation")
    check_file_existence(df_test, "Test")

    # 3. Validation Logic Verification
    print("\nVerifying split logic...")

    # Check overlap
    train_ids = set(df_train["image_id"])
    val_ids = set(df_val["image_id"])
    overlap = train_ids.intersection(val_ids)
    assert (
        len(overlap) == 0
    ), f"Found {len(overlap)} overlapping IDs between train and val sets."

    # Check Split Ratio
    total_train_val = len(df_train) + len(df_val)
    val_ratio = len(df_val) / total_train_val
    print(f"Actual Validation Ratio: {val_ratio:.4f}")

    # Allow small deviation due to discrete counts
    assert (
        0.19 <= val_ratio <= 0.21
    ), f"Validation ratio {val_ratio} is not close to 0.2"

    # Check Stratification
    train_dist = df_train["label"].value_counts(normalize=True).sort_index()
    val_dist = df_val["label"].value_counts(normalize=True).sort_index()

    # Calculate max difference in class proportions
    diffs = (train_dist - val_dist).abs()
    max_diff = diffs.max()
    print(f"Max class distribution difference: {max_diff:.4f}")

    # Assert stratification is reasonably close (e.g., within 1-2%)
    if max_diff > 0.02:
        raise AssertionError(
            f"Stratification failed. Max difference in class distribution is {max_diff}"
        )

    print("\nAll validation checks passed successfully.")


if __name__ == "__main__":
    try:
        train_p, val_p, test_p = generate_metadata()
        validate_metadata(train_p, val_p, test_p)
    except Exception as e:
        print(f"\nERROR: {e}")
        exit(1)
