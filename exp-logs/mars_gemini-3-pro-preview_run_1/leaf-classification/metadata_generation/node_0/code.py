import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Constants
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
IMAGES_DIR_NAME = "images"
RANDOM_STATE = 42
VAL_SIZE = 0.2


def generate_metadata():
    # Ensure metadata directory exists
    os.makedirs(METADATA_DIR, exist_ok=True)

    # Load raw data
    print("Loading raw data...")
    train_path = os.path.join(INPUT_DIR, "train.csv")
    test_path = os.path.join(INPUT_DIR, "test.csv")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"train.csv not found at {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"test.csv not found at {test_path}")

    df_train_full = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    # Construct relative file paths for images
    # Image files are named with their id (e.g., 123.jpg)
    def get_image_path(row):
        return os.path.join(IMAGES_DIR_NAME, f"{int(row['id'])}.jpg")

    df_train_full["file_path"] = df_train_full.apply(get_image_path, axis=1)
    df_test["file_path"] = df_test.apply(get_image_path, axis=1)

    # Check if 'species' column exists for stratification
    if "species" not in df_train_full.columns:
        # In some versions of this dataset, the target might be named differently or implied.
        # However, typically for this specific task, it is 'species'.
        # If missing, we cannot perform stratified sampling on the target.
        raise ValueError(
            "Column 'species' not found in train.csv. Cannot perform stratified split."
        )

    # Perform Stratified Split
    print("Splitting training data into train and validation sets...")
    try:
        df_train, df_val = train_test_split(
            df_train_full,
            test_size=VAL_SIZE,
            random_state=RANDOM_STATE,
            stratify=df_train_full["species"],
        )
    except ValueError as e:
        # This might happen if a class has too few members (< 2)
        print(f"Error during stratified split: {e}")
        print(
            "Attempting to handle rare classes or falling back to random split if strictly necessary."
        )
        # Check class counts
        class_counts = df_train_full["species"].value_counts()
        rare_classes = class_counts[class_counts < 2].index.tolist()
        if rare_classes:
            print(
                f"Warning: The following classes have fewer than 2 samples: {rare_classes}"
            )
            raise ValueError(
                f"Cannot perform stratified split. Classes {rare_classes} have insufficient samples."
            )
        raise e

    # Save metadata
    print("Saving metadata...")
    train_meta_path = os.path.join(METADATA_DIR, "train.csv")
    val_meta_path = os.path.join(METADATA_DIR, "val.csv")
    test_meta_path = os.path.join(METADATA_DIR, "test.csv")

    df_train.to_csv(train_meta_path, index=False)
    df_val.to_csv(val_meta_path, index=False)
    df_test.to_csv(test_meta_path, index=False)

    print(f"Metadata saved to {METADATA_DIR}")
    return train_meta_path, val_meta_path, test_meta_path


def validate_metadata(train_path, val_path, test_path):
    print("\nStarting validation...")

    # Load generated metadata
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    # 1. Print Summary Statistics
    print("\n--- Summary Statistics ---")
    print(f"Train set shape: {df_train.shape}")
    print(f"Val set shape:   {df_val.shape}")
    print(f"Test set shape:  {df_test.shape}")

    print(f"Train unique species: {df_train['species'].nunique()}")
    print(f"Val unique species:   {df_val['species'].nunique()}")

    # 2. Check File Paths
    print("\n--- Checking File Paths ---")
    datasets = {"train": df_train, "val": df_val, "test": df_test}

    for name, df in datasets.items():
        print(f"Checking {name} paths...")
        # Randomly select up to 1000 paths
        sample_size = min(1000, len(df))
        sample_paths = df["file_path"].sample(n=sample_size, random_state=RANDOM_STATE)

        missing_count = 0
        missing_samples = []

        for rel_path in sample_paths:
            full_path = os.path.join(INPUT_DIR, rel_path)
            if not os.path.exists(full_path):
                missing_count += 1
                if len(missing_samples) < 5:
                    missing_samples.append(rel_path)

        missing_ratio = missing_count / sample_size
        print(
            f"  Missing file ratio for {name}: {missing_ratio:.4f} ({missing_count}/{sample_size})"
        )

        if missing_ratio > 0.5:
            print("  Sample missing paths:", missing_samples)
            raise FileNotFoundError(
                f"More than 50% of file paths in {name} metadata do not exist in {INPUT_DIR}."
            )

    # 3. Verify Validation Split Requirements
    print("\n--- Verifying Split Requirements ---")

    # Check split ratio
    total_train_val = len(df_train) + len(df_val)
    val_ratio = len(df_val) / total_train_val
    print(f"Actual Validation Ratio: {val_ratio:.4f} (Target: {VAL_SIZE})")

    # Allow a small margin of error due to integer division/rounding
    assert (
        0.19 < val_ratio < 0.21
    ), f"Validation split ratio {val_ratio} deviates significantly from 0.2"

    # Check overlap
    train_ids = set(df_train["id"])
    val_ids = set(df_val["id"])
    overlap = train_ids.intersection(val_ids)
    assert (
        len(overlap) == 0
    ), f"Found {len(overlap)} overlapping IDs between train and val sets."

    # Check Stratification
    # We compare the distribution of species in train and val
    train_dist = df_train["species"].value_counts(normalize=True).sort_index()
    val_dist = df_val["species"].value_counts(normalize=True).sort_index()

    # Align indices to ensure we are comparing same species
    all_species = sorted(list(set(train_dist.index) | set(val_dist.index)))
    train_dist = train_dist.reindex(all_species, fill_value=0)
    val_dist = val_dist.reindex(all_species, fill_value=0)

    # Calculate Maximum Absolute Difference in proportions
    max_diff = (train_dist - val_dist).abs().max()
    print(f"Max difference in class proportions between Train and Val: {max_diff:.4f}")

    # Threshold for stratification failure.
    # With small datasets and many classes, perfect stratification is hard, but diff should be small.
    # 0.05 is a generous buffer, but sufficient to catch random splitting vs stratified.
    if max_diff > 0.05:
        print("Top differences:")
        print((train_dist - val_dist).abs().sort_values(ascending=False).head())
        raise AssertionError(
            "Stratification check failed: Class distributions differ significantly."
        )

    print("\nValidation successful! Metadata is ready.")


if __name__ == "__main__":
    try:
        train_csv, val_csv, test_csv = generate_metadata()
        validate_metadata(train_csv, val_csv, test_csv)
    except Exception as e:
        print(f"\nFAILED: {e}")
        exit(1)
