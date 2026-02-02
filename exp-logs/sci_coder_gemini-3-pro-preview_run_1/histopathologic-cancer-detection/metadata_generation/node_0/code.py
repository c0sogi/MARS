import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def generate_metadata():
    # Constants
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    TRAIN_DIR_NAME = "train"
    TEST_DIR_NAME = "test"
    RANDOM_STATE = 42
    VAL_SIZE = 0.2

    # Create metadata directory
    os.makedirs(METADATA_DIR, exist_ok=True)

    print("Generating metadata...")

    # 1. Process Training Data
    train_labels_path = os.path.join(INPUT_DIR, "train_labels.csv")
    if not os.path.exists(train_labels_path):
        raise FileNotFoundError(f"Could not find {train_labels_path}")

    df_full_train = pd.read_csv(train_labels_path)

    # Construct relative file paths for training images
    # Assuming files are .tif based on dataset description
    df_full_train["file_path"] = df_full_train["id"].apply(
        lambda x: os.path.join(TRAIN_DIR_NAME, f"{x}.tif")
    )

    # Split into Train and Validation
    # Using stratified split because it is a classification task
    train_df, val_df = train_test_split(
        df_full_train,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=df_full_train["label"],
    )

    # 2. Process Test Data
    # Scan the test directory to ensure we get actual files
    test_dir_path = os.path.join(INPUT_DIR, TEST_DIR_NAME)
    if not os.path.exists(test_dir_path):
        raise FileNotFoundError(f"Could not find {test_dir_path}")

    test_files = glob.glob(os.path.join(test_dir_path, "*.tif"))

    test_data = []
    for filepath in test_files:
        # Extract filename and id
        filename = os.path.basename(filepath)
        img_id = os.path.splitext(filename)[0]
        # Relative path from input dir
        rel_path = os.path.join(TEST_DIR_NAME, filename)
        test_data.append({"id": img_id, "file_path": rel_path})

    test_df = pd.DataFrame(test_data)

    # 3. Save Metadata
    train_save_path = os.path.join(METADATA_DIR, "train.csv")
    val_save_path = os.path.join(METADATA_DIR, "val.csv")
    test_save_path = os.path.join(METADATA_DIR, "test.csv")

    train_df.to_csv(train_save_path, index=False)
    val_df.to_csv(val_save_path, index=False)
    test_df.to_csv(test_save_path, index=False)

    print(f"Saved train metadata to {train_save_path}")
    print(f"Saved val metadata to {val_save_path}")
    print(f"Saved test metadata to {test_save_path}")

    return train_save_path, val_save_path, test_save_path


def verify_metadata(train_path, val_path, test_path):
    print("\nVerifying metadata...")
    INPUT_DIR = "./input"

    # Load datasets
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    # 1. Print Summary Statistics
    print("-" * 30)
    print("Summary Statistics")
    print("-" * 30)
    print(f"Train set shape: {train_df.shape}")
    print(
        f"Train label distribution:\n{train_df['label'].value_counts(normalize=True)}"
    )
    print(f"\nValidation set shape: {val_df.shape}")
    print(
        f"Validation label distribution:\n{val_df['label'].value_counts(normalize=True)}"
    )
    print(f"\nTest set shape: {test_df.shape}")
    print("-" * 30)

    # 2. Check File Paths
    datasets = [("Train", train_df), ("Validation", val_df), ("Test", test_df)]

    for name, df in datasets:
        print(f"Checking file paths for {name} dataset...")
        # Sample 1000 paths or all if less than 1000
        n_samples = min(1000, len(df))
        sample_paths = df["file_path"].sample(n=n_samples, random_state=42).tolist()

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
            f"  Missing file ratio: {missing_ratio:.4f} ({missing_count}/{n_samples})"
        )

        if missing_ratio > 0.5:
            print("  Sample missing paths:")
            for p in missing_samples:
                print(f"    {p}")
            raise FileNotFoundError(
                f"More than 50% of file paths in {name} metadata do not resolve."
            )

    # 3. Verify Validation Split Requirements
    print("\nVerifying validation split requirements...")

    # Check 1: No overlap between train and val IDs
    train_ids = set(train_df["id"])
    val_ids = set(val_df["id"])
    overlap = train_ids.intersection(val_ids)
    if len(overlap) > 0:
        raise AssertionError(
            f"Found {len(overlap)} overlapping IDs between train and validation sets."
        )
    print("  PASS: No ID overlap between train and validation.")

    # Check 2: Stratification
    # Compare label proportions
    train_pos_ratio = train_df["label"].mean()
    val_pos_ratio = val_df["label"].mean()
    diff = abs(train_pos_ratio - val_pos_ratio)

    print(f"  Train positive ratio: {train_pos_ratio:.4f}")
    print(f"  Val positive ratio:   {val_pos_ratio:.4f}")
    print(f"  Difference:           {diff:.4f}")

    # Allow a small tolerance for stratification differences (e.g., 1%)
    if diff > 0.01:
        raise AssertionError(
            "Stratification check failed. Label distributions differ significantly."
        )
    print("  PASS: Stratification check passed.")

    print("\nAll verification checks passed successfully.")


if __name__ == "__main__":
    try:
        train_p, val_p, test_p = generate_metadata()
        verify_metadata(train_p, val_p, test_p)
    except Exception as e:
        print(f"\nERROR: {e}")
        exit(1)
