import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Constants
INPUT_DIR = "./input"
TRAIN_DIR = os.path.join(INPUT_DIR, "train")
TEST_DIR = os.path.join(INPUT_DIR, "test")
METADATA_DIR = "./metadata"
RANDOM_STATE = 42


def generate_metadata():
    # Ensure metadata directory exists
    os.makedirs(METADATA_DIR, exist_ok=True)

    print("Scanning training directory...")
    # Process Training Data
    train_files = os.listdir(TRAIN_DIR)
    train_data = []

    for filename in train_files:
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        # Filename format: cat.0.jpg or dog.1.jpg
        # We assume the label is the first part of the filename
        parts = filename.split(".")
        label_str = parts[0].lower()

        # Assign label: 1 for dog, 0 for cat
        if label_str == "dog":
            label = 1
        elif label_str == "cat":
            label = 0
        else:
            # Skip files that don't match the expected pattern
            continue

        train_data.append({"filepath": os.path.join("train", filename), "label": label})

    full_train_df = pd.DataFrame(train_data)

    # Split into Train and Validation (80:20, Stratified)
    print("Splitting data into train and validation sets...")
    train_df, val_df = train_test_split(
        full_train_df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=full_train_df["label"],
    )

    # Process Test Data
    print("Scanning test directory...")
    test_files = os.listdir(TEST_DIR)
    test_data = []

    for filename in test_files:
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        # Filename format: 1.jpg
        try:
            img_id = int(filename.split(".")[0])
        except ValueError:
            continue

        test_data.append({"filepath": os.path.join("test", filename), "id": img_id})

    test_df = pd.DataFrame(test_data)
    # Sort by ID for consistency
    test_df = test_df.sort_values("id").reset_index(drop=True)

    # Save Metadata
    print("Saving metadata files...")
    train_df.to_csv(os.path.join(METADATA_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(METADATA_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(METADATA_DIR, "test.csv"), index=False)

    return train_df, val_df, test_df


def validate_metadata(train_df, val_df, test_df):
    print("\n=== Validation & Summary ===")

    datasets = {"Train": train_df, "Validation": val_df, "Test": test_df}

    # 1. Print Summary Statistics
    for name, df in datasets.items():
        print(f"\n{name} Dataset:")
        print(f"  Total samples: {len(df)}")
        if "label" in df.columns:
            print(f"  Class distribution:\n{df['label'].value_counts(normalize=True)}")
        else:
            print(f"  Columns: {list(df.columns)}")

    # 2. Check File Paths
    print("\nChecking file path existence...")
    for name, df in datasets.items():
        # Select up to 1000 random samples
        n_samples = min(1000, len(df))
        if n_samples == 0:
            continue

        sample_df = df.sample(n=n_samples, random_state=RANDOM_STATE)
        missing_count = 0
        missing_samples = []

        for _, row in sample_df.iterrows():
            # Relative path from metadata needs to be joined with INPUT_DIR to resolve
            full_path = os.path.join(INPUT_DIR, row["filepath"])
            if not os.path.exists(full_path):
                missing_count += 1
                if len(missing_samples) < 5:
                    missing_samples.append(row["filepath"])

        missing_ratio = missing_count / n_samples
        print(
            f"  {name}: Missing ratio = {missing_ratio:.4f} ({missing_count}/{n_samples})"
        )

        if missing_ratio > 0.5:
            print(f"  Sample missing paths: {missing_samples}")
            raise FileNotFoundError(
                f"More than 50% of files missing in {name} dataset."
            )

    # 3. Verify Stratification
    print("\nVerifying stratification...")
    train_dist = train_df["label"].value_counts(normalize=True)
    val_dist = val_df["label"].value_counts(normalize=True)

    print(f"  Train class 1 ratio: {train_dist.get(1, 0):.4f}")
    print(f"  Val class 1 ratio:   {val_dist.get(1, 0):.4f}")

    # Check if the difference in distribution is within a small tolerance (e.g., 1%)
    # Note: With small datasets, exact matches aren't always possible, but stratify should keep it close.
    diff = abs(train_dist.get(1, 0) - val_dist.get(1, 0))
    if diff > 0.01:
        raise AssertionError(
            f"Stratification failed. Class distribution difference {diff:.4f} exceeds tolerance."
        )

    print("Stratification check passed.")
    print("\nMetadata generation and validation completed successfully.")


if __name__ == "__main__":
    train_df, val_df, test_df = generate_metadata()

    # Reload datasets from disk to ensure the saved files are correct
    train_df_loaded = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
    val_df_loaded = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))
    test_df_loaded = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))

    validate_metadata(train_df_loaded, val_df_loaded, test_df_loaded)
