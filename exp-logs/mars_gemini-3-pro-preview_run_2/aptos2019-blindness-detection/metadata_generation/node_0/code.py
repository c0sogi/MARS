import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Configuration
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
RANDOM_STATE = 42
VAL_SIZE = 0.2


def generate_metadata():
    print("Starting metadata generation...")

    # Ensure metadata directory exists
    os.makedirs(METADATA_DIR, exist_ok=True)

    # Load raw csv files
    train_csv_path = os.path.join(INPUT_DIR, "train.csv")
    test_csv_path = os.path.join(INPUT_DIR, "test.csv")

    df_train_full = pd.read_csv(train_csv_path)
    df_test = pd.read_csv(test_csv_path)

    print(f"Loaded train.csv with {len(df_train_full)} rows.")
    print(f"Loaded test.csv with {len(df_test)} rows.")

    # Construct relative file paths
    # Images are in train_images/ and test_images/ with .png extension based on id_code
    df_train_full["file_path"] = df_train_full["id_code"].apply(
        lambda x: os.path.join("train_images", f"{x}.png")
    )
    df_test["file_path"] = df_test["id_code"].apply(
        lambda x: os.path.join("test_images", f"{x}.png")
    )

    # Split train into train/val using stratified sampling
    # We stratify on 'diagnosis'
    print("Splitting training data into train and validation sets...")
    df_train, df_val = train_test_split(
        df_train_full,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=df_train_full["diagnosis"],
    )

    # Save metadata files
    train_save_path = os.path.join(METADATA_DIR, "train.csv")
    val_save_path = os.path.join(METADATA_DIR, "val.csv")
    test_save_path = os.path.join(METADATA_DIR, "test.csv")

    df_train.to_csv(train_save_path, index=False)
    df_val.to_csv(val_save_path, index=False)
    df_test.to_csv(test_save_path, index=False)

    print(f"Saved metadata to {METADATA_DIR}")


def verify_metadata():
    print("\nStarting metadata verification...")

    # Load generated metadata
    df_train = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
    df_val = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))
    df_test = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))

    # 1. Print Summary Statistics
    print("\n=== Summary Statistics ===")
    print(f"Train set size: {len(df_train)}")
    print(f"Val set size: {len(df_val)}")
    print(f"Test set size: {len(df_test)}")

    print("\nTrain Class Distribution:")
    print(df_train["diagnosis"].value_counts(normalize=True).sort_index())

    print("\nVal Class Distribution:")
    print(df_val["diagnosis"].value_counts(normalize=True).sort_index())

    # 2. Check File Paths
    print("\n=== Checking File Paths ===")
    datasets = {"train": df_train, "val": df_val, "test": df_test}

    for name, df in datasets.items():
        # Select up to 1000 random paths
        n_samples = min(1000, len(df))
        sample_paths = df["file_path"].sample(n=n_samples, random_state=RANDOM_STATE)

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
            f"Dataset '{name}': Checked {n_samples} files. Missing ratio: {missing_ratio:.4f}"
        )

        if missing_ratio > 0.5:
            print("Sample missing paths:", missing_samples)
            raise FileNotFoundError(
                f"More than 50% of files missing in {name} dataset."
            )

    # 3. Verify Stratification
    print("\n=== Verifying Stratification ===")
    train_dist = df_train["diagnosis"].value_counts(normalize=True).sort_index()
    val_dist = df_val["diagnosis"].value_counts(normalize=True).sort_index()

    # Check if distributions are similar (within 5% tolerance per class)
    diff = (train_dist - val_dist).abs()
    print("Difference in class distribution (Train - Val):")
    print(diff)

    if (diff > 0.05).any():
        raise AssertionError(
            "Stratification failed: Class distributions differ significantly between train and val."
        )

    print("Stratification check passed.")

    # Verify split ratio
    total_train_val = len(df_train) + len(df_val)
    actual_val_ratio = len(df_val) / total_train_val
    print(f"Actual validation ratio: {actual_val_ratio:.4f} (Target: {VAL_SIZE})")

    if not np.isclose(actual_val_ratio, VAL_SIZE, atol=0.01):
        raise AssertionError(
            f"Split ratio mismatch. Expected {VAL_SIZE}, got {actual_val_ratio:.4f}"
        )

    print("\nAll verification checks passed successfully.")


if __name__ == "__main__":
    generate_metadata()
    verify_metadata()
