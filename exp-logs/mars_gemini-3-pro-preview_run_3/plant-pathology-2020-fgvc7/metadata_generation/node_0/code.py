import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Configuration
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
IMAGES_DIR = "images"
RANDOM_STATE = 42
VAL_SIZE = 0.2


def generate_metadata():
    """
    Generates metadata CSVs for train, val, and test sets.
    """
    # Create metadata directory
    os.makedirs(METADATA_DIR, exist_ok=True)

    # Load raw csv files
    train_path = os.path.join(INPUT_DIR, "train.csv")
    test_path = os.path.join(INPUT_DIR, "test.csv")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"{train_path} not found.")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"{test_path} not found.")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Identify label columns (all columns except image_id)
    label_cols = [col for col in train_df.columns if col != "image_id"]

    # Construct relative file paths
    # Based on dataset info, image_id does not have extension, but files are .jpg
    train_df["file_path"] = train_df["image_id"].apply(
        lambda x: os.path.join(IMAGES_DIR, f"{x}.jpg")
    )
    test_df["file_path"] = test_df["image_id"].apply(
        lambda x: os.path.join(IMAGES_DIR, f"{x}.jpg")
    )

    # Create a single label column for stratification
    # The problem defines mutually exclusive classes (including 'multiple_diseases')
    # We use idxmax to get the active class for each row
    train_df["stratify_label"] = train_df[label_cols].idxmax(axis=1)

    # Perform stratified split
    train_split, val_split = train_test_split(
        train_df,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=train_df["stratify_label"],
        shuffle=True,
    )

    # Save metadata files
    train_split.to_csv(os.path.join(METADATA_DIR, "train.csv"), index=False)
    val_split.to_csv(os.path.join(METADATA_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(METADATA_DIR, "test.csv"), index=False)

    print("Metadata generation complete.")
    return label_cols


def check_file_paths(df, name):
    """
    Checks if a sample of file paths exist.
    """
    sample_size = min(1000, len(df))
    sample = df["file_path"].sample(n=sample_size, random_state=RANDOM_STATE)

    missing_files = []
    for rel_path in sample:
        full_path = os.path.join(INPUT_DIR, rel_path)
        if not os.path.exists(full_path):
            missing_files.append(rel_path)

    missing_ratio = len(missing_files) / sample_size
    print(
        f"[{name}] Missing file ratio: {missing_ratio:.4f} ({len(missing_files)}/{sample_size})"
    )

    if missing_ratio > 0.5:
        print(f"Sample of missing files in {name}: {missing_files[:5]}")
        raise FileNotFoundError(f"More than 50% of files missing in {name} dataset.")


def validate_metadata(label_cols):
    """
    Loads generated metadata and performs validation checks.
    """
    print("\nStarting validation...")

    # Load generated metadata
    train_df = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))
    test_df = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))

    # 1. Print Summary Statistics
    print("\nSummary Statistics:")
    print("-" * 30)
    print(f"Train set size: {len(train_df)}")
    print(f"Val set size:   {len(val_df)}")
    print(f"Test set size:  {len(test_df)}")

    print("\nTrain Class Distribution:")
    print(train_df["stratify_label"].value_counts(normalize=True))
    print("\nVal Class Distribution:")
    print(val_df["stratify_label"].value_counts(normalize=True))

    # 2. Check File Paths
    print("\nChecking file paths...")
    check_file_paths(train_df, "train")
    check_file_paths(val_df, "val")
    check_file_paths(test_df, "test")

    # 3. Verify Stratification
    print("\nVerifying stratification...")
    train_dist = train_df["stratify_label"].value_counts(normalize=True)
    val_dist = val_df["stratify_label"].value_counts(normalize=True)

    # Check if distributions are similar (within 5% tolerance)
    for label in train_dist.index:
        train_prop = train_dist[label]
        val_prop = val_dist.get(label, 0)
        diff = abs(train_prop - val_prop)

        if diff > 0.05:
            raise AssertionError(
                f"Stratification check failed for class '{label}'. "
                f"Train prop: {train_prop:.4f}, Val prop: {val_prop:.4f}, Diff: {diff:.4f}"
            )

    print("Stratification verification passed.")
    print("\nAll checks passed successfully.")


if __name__ == "__main__":
    try:
        labels = generate_metadata()
        validate_metadata(labels)
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        raise e
