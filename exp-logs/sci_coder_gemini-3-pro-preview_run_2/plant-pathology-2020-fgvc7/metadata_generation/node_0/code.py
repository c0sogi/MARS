import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Constants
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
IMAGES_DIR = "images"
RANDOM_STATE = 42
VAL_SIZE = 0.2


def generate_metadata():
    """
    Reads raw data, performs stratified split, and saves metadata files.
    """
    # Create metadata directory
    os.makedirs(METADATA_DIR, exist_ok=True)

    # --- Process Training Data ---
    train_path = os.path.join(INPUT_DIR, "train.csv")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"train.csv not found at {train_path}")

    train_df = pd.read_csv(train_path)

    # Identify label columns (all columns except image_id)
    label_cols = [c for c in train_df.columns if c != "image_id"]

    # Construct relative file paths
    # Assuming format: images/{image_id}.jpg
    train_df["file_path"] = train_df["image_id"].apply(
        lambda x: os.path.join(IMAGES_DIR, f"{x}.jpg")
    )

    # Create a single label column for stratification
    # We assume the task is multi-class (Healthy, Rust, Scab, Multiple) based on description
    # idxmax will return the column name with the highest value (1.0 in one-hot)
    train_df["stratify_label"] = train_df[label_cols].idxmax(axis=1)

    # Perform Stratified Split
    train_meta, val_meta = train_test_split(
        train_df,
        test_size=VAL_SIZE,
        stratify=train_df["stratify_label"],
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    # Save Train and Validation Metadata
    train_meta_path = os.path.join(METADATA_DIR, "train_metadata.csv")
    val_meta_path = os.path.join(METADATA_DIR, "val_metadata.csv")

    train_meta.to_csv(train_meta_path, index=False)
    val_meta.to_csv(val_meta_path, index=False)

    # --- Process Test Data ---
    test_path = os.path.join(INPUT_DIR, "test.csv")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"test.csv not found at {test_path}")

    test_df = pd.read_csv(test_path)
    test_df["file_path"] = test_df["image_id"].apply(
        lambda x: os.path.join(IMAGES_DIR, f"{x}.jpg")
    )

    test_meta_path = os.path.join(METADATA_DIR, "test_metadata.csv")
    test_df.to_csv(test_meta_path, index=False)

    print("Metadata generation complete.")
    return label_cols


def verify_metadata(label_cols):
    """
    Loads generated metadata and performs validation checks.
    """
    print("\nStarting verification...")

    # Load datasets
    train_meta = pd.read_csv(os.path.join(METADATA_DIR, "train_metadata.csv"))
    val_meta = pd.read_csv(os.path.join(METADATA_DIR, "val_metadata.csv"))
    test_meta = pd.read_csv(os.path.join(METADATA_DIR, "test_metadata.csv"))

    # 1. Print Summary Statistics
    print("-" * 30)
    print(f"Train set shape: {train_meta.shape}")
    print(f"Val set shape:   {val_meta.shape}")
    print(f"Test set shape:  {test_meta.shape}")
    print("-" * 30)

    print("Train Class Distribution:")
    train_dist = train_meta["stratify_label"].value_counts(normalize=True)
    print(train_dist)

    print("\nValidation Class Distribution:")
    val_dist = val_meta["stratify_label"].value_counts(normalize=True)
    print(val_dist)
    print("-" * 30)

    # 2. Check File Paths
    def check_file_existence(df, dataset_name):
        # Check up to 1000 random paths
        sample_size = min(1000, len(df))
        sample_paths = (
            df["file_path"].sample(n=sample_size, random_state=RANDOM_STATE).tolist()
        )

        missing_files = []
        for rel_path in sample_paths:
            full_path = os.path.join(INPUT_DIR, rel_path)
            if not os.path.exists(full_path):
                missing_files.append(rel_path)

        missing_ratio = len(missing_files) / sample_size
        print(f"[{dataset_name}] Missing file ratio: {missing_ratio:.4f}")

        if missing_ratio > 0.5:
            print(f"Sample missing files: {missing_files[:5]}")
            raise FileNotFoundError(
                f"Error: More than 50% of files are missing in {dataset_name} dataset."
            )

    check_file_existence(train_meta, "Train")
    check_file_existence(val_meta, "Validation")
    check_file_existence(test_meta, "Test")

    # 3. Verify Stratification
    # We expect the distribution of classes in Train and Val to be very similar.
    # We'll check that the difference in proportions for each class is within a small tolerance (e.g., 2%).
    print("\nVerifying Stratification...")

    # Ensure all classes in train are in val (unless class count is extremely low, which shouldn't happen here)
    train_classes = set(train_dist.index)
    val_classes = set(val_dist.index)

    if train_classes != val_classes:
        raise AssertionError(
            f"Class mismatch between Train and Val sets.\nTrain: {train_classes}\nVal: {val_classes}"
        )

    for label in train_classes:
        train_prop = train_dist[label]
        val_prop = val_dist[label]
        diff = abs(train_prop - val_prop)

        # Tolerance: 0.05 (5%) is generous but accounts for small validation sets where 1 sample changes % significantly
        if diff > 0.05:
            raise AssertionError(
                f"Stratification failed for class '{label}'. Train prop: {train_prop:.4f}, Val prop: {val_prop:.4f}, Diff: {diff:.4f}"
            )

    print("Stratification verification passed.")
    print("\nAll checks passed successfully.")


if __name__ == "__main__":
    try:
        labels = generate_metadata()
        verify_metadata(labels)
    except Exception as e:
        print(f"\nScript failed with error: {e}")
        exit(1)
