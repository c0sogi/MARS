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
    # 1. Setup Directories
    os.makedirs(METADATA_DIR, exist_ok=True)

    # 2. Load Raw Data
    print("Loading raw data...")
    train_path = os.path.join(INPUT_DIR, "train.csv")
    test_path = os.path.join(INPUT_DIR, "test.csv")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(
            "Could not find train.csv or test.csv in input directory."
        )

    df_train_full = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    # 3. Add Image Paths (Relative to input directory)
    # Format: images/{id}.jpg
    df_train_full["image_path"] = df_train_full["id"].apply(
        lambda x: os.path.join(IMAGES_DIR, f"{x}.jpg")
    )
    df_test["image_path"] = df_test["id"].apply(
        lambda x: os.path.join(IMAGES_DIR, f"{x}.jpg")
    )

    # 4. Create Validation Split
    print("Splitting data into train and validation sets...")
    # Using stratified split to maintain species distribution
    # Note: If a class has only 1 sample, stratify will fail.
    # Given the dataset description (Leaf Classification), classes typically have multiple samples.
    try:
        df_train, df_val = train_test_split(
            df_train_full,
            test_size=VAL_SIZE,
            random_state=RANDOM_STATE,
            shuffle=True,
            stratify=df_train_full["species"],
        )
    except ValueError as e:
        # Fallback if stratification fails (e.g. single sample classes), though unlikely for this dataset
        print(f"Warning: Stratified split failed ({e}). Falling back to random split.")
        df_train, df_val = train_test_split(
            df_train_full, test_size=VAL_SIZE, random_state=RANDOM_STATE, shuffle=True
        )

    # 5. Save Metadata
    print("Saving metadata...")
    df_train.to_csv(os.path.join(METADATA_DIR, "train.csv"), index=False)
    df_val.to_csv(os.path.join(METADATA_DIR, "val.csv"), index=False)
    df_test.to_csv(os.path.join(METADATA_DIR, "test.csv"), index=False)

    print("Metadata generation complete.")


def verify_metadata():
    print("\nStarting verification...")

    # Load generated metadata
    df_train = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
    df_val = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))
    df_test = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))

    # 1. Summary Statistics
    print("-" * 30)
    print("Summary Statistics:")
    print(f"Train set shape: {df_train.shape}")
    print(f"Val set shape:   {df_val.shape}")
    print(f"Test set shape:  {df_test.shape}")
    print(f"Unique species in Train: {df_train['species'].nunique()}")
    print(f"Unique species in Val:   {df_val['species'].nunique()}")
    print("-" * 30)

    # 2. File Path Check
    print("Checking file paths...")
    datasets = {"train": df_train, "val": df_val, "test": df_test}

    for name, df in datasets.items():
        # Sample 1000 paths or all if less than 1000
        n_samples = min(1000, len(df))
        sample_paths = (
            df["image_path"].sample(n=n_samples, random_state=RANDOM_STATE).values
        )

        missing_count = 0
        missing_samples = []

        for rel_path in sample_paths:
            full_path = os.path.join(INPUT_DIR, rel_path)
            if not os.path.exists(full_path):
                missing_count += 1
                if len(missing_samples) < 5:
                    missing_samples.append(rel_path)

        missing_ratio = missing_count / n_samples
        print(
            f"[{name}] Missing file ratio: {missing_ratio:.4f} ({missing_count}/{n_samples})"
        )

        if missing_ratio > 0.5:
            print(f"Sample missing paths from {name}: {missing_samples}")
            raise FileNotFoundError(
                f"More than 50% of image files are missing in {name} dataset."
            )

    # 3. Verify Validation Split Requirements
    print("Verifying split requirements...")

    # Check split ratio
    total_train_val = len(df_train) + len(df_val)
    val_ratio = len(df_val) / total_train_val
    print(f"Actual Validation Ratio: {val_ratio:.4f}")

    # Allow small floating point deviation
    assert (
        0.19 <= val_ratio <= 0.21
    ), f"Validation split ratio {val_ratio} deviates significantly from 0.2"

    # Check stratification (compare class distributions)
    # We calculate the normalized value counts for both sets
    train_dist = df_train["species"].value_counts(normalize=True).sort_index()
    val_dist = df_val["species"].value_counts(normalize=True).sort_index()

    # Align indices to ensure we are comparing same classes (handle case where val might miss a rare class)
    all_classes = sorted(list(set(train_dist.index) | set(val_dist.index)))
    train_dist = train_dist.reindex(all_classes, fill_value=0)
    val_dist = val_dist.reindex(all_classes, fill_value=0)

    # Calculate Mean Absolute Error between distributions
    dist_diff = np.abs(train_dist - val_dist).mean()
    print(f"Mean absolute difference in class distribution: {dist_diff:.6f}")

    # Threshold: If stratification worked, distributions should be very close.
    # Given small dataset size (N~900, 99 classes), some noise is expected.
    # 0.01 is a reasonable loose bound for this specific dataset size/class count.
    if dist_diff > 0.02:
        raise AssertionError(
            "Stratification check failed: Class distributions differ significantly."
        )

    print("All checks passed successfully.")


if __name__ == "__main__":
    generate_metadata()
    verify_metadata()
