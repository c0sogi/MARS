import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Constants
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
RANDOM_STATE = 42
VAL_SIZE = 0.2


def generate_metadata():
    print("Starting metadata generation...")

    # Create metadata directory
    os.makedirs(METADATA_DIR, exist_ok=True)

    # 1. Load Raw Data
    labels_path = os.path.join(INPUT_DIR, "labels.csv")
    sample_sub_path = os.path.join(INPUT_DIR, "sample_submission.csv")

    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"labels.csv not found at {labels_path}")

    # Load labels (Training data source)
    df_full_train = pd.read_csv(labels_path)

    # Load sample submission (Test data source)
    # If sample_submission doesn't exist, we might need to list files,
    # but the prompt guarantees its existence.
    if os.path.exists(sample_sub_path):
        df_test = pd.read_csv(sample_sub_path)
    else:
        # Fallback: list files in test directory if sample_submission is missing
        test_dir = os.path.join(INPUT_DIR, "test")
        if os.path.exists(test_dir):
            test_files = [f for f in os.listdir(test_dir) if f.endswith(".jpg")]
            # Remove extension for ID
            test_ids = [os.path.splitext(f)[0] for f in test_files]
            df_test = pd.DataFrame({"id": test_ids})
        else:
            raise FileNotFoundError("Could not locate test data source.")

    # 2. Construct File Paths
    # Paths must be relative to ./input
    # Training images are in input/train/{id}.jpg
    # Test images are in input/test/{id}.jpg

    df_full_train["file_path"] = df_full_train["id"].apply(
        lambda x: os.path.join("train", f"{x}.jpg")
    )
    df_test["file_path"] = df_test["id"].apply(
        lambda x: os.path.join("test", f"{x}.jpg")
    )

    # 3. Split Training Data (Stratified)
    print(f"Splitting data with ratio 80:20, random_state={RANDOM_STATE}...")

    # Check if we can stratify (ensure at least 2 samples per class)
    class_counts = df_full_train["breed"].value_counts()
    single_sample_classes = class_counts[class_counts < 2].index.tolist()

    if single_sample_classes:
        print(
            f"Warning: The following classes have < 2 samples and cannot be stratified: {single_sample_classes}"
        )
        # In a strict scenario, we might drop them or just do random split.
        # For this dataset, usually all classes have sufficient samples.
        # We will proceed with standard stratification which will error if < 2 members.

    train_df, val_df = train_test_split(
        df_full_train,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=df_full_train["breed"],
    )

    # 4. Save Metadata
    train_csv_path = os.path.join(METADATA_DIR, "train.csv")
    val_csv_path = os.path.join(METADATA_DIR, "val.csv")
    test_csv_path = os.path.join(METADATA_DIR, "test.csv")

    train_df.to_csv(train_csv_path, index=False)
    val_df.to_csv(val_csv_path, index=False)
    df_test.to_csv(test_csv_path, index=False)

    print("Metadata generation complete.")
    return train_csv_path, val_csv_path, test_csv_path


def validate_metadata(train_path, val_path, test_path):
    print("\nStarting validation of generated metadata...")

    # Load datasets
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    # 1. Summary Statistics
    print("\nSummary Statistics:")
    print(f"Train set shape: {df_train.shape}")
    print(f"Validation set shape: {df_val.shape}")
    print(f"Test set shape: {df_test.shape}")

    print(f"Train unique breeds: {df_train['breed'].nunique()}")
    print(f"Validation unique breeds: {df_val['breed'].nunique()}")

    # 2. Check File Existence
    def check_paths(df, name):
        print(f"\nChecking file paths for {name} set...")
        # Sample 1000 or all if less
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
        print(f"Missing file ratio: {missing_ratio:.4f} ({missing_count}/{n_samples})")

        if missing_ratio > 0.5:
            print("Sample missing paths:", missing_samples)
            raise FileNotFoundError(
                f"Missing file ratio for {name} is {missing_ratio}, which exceeds 0.5"
            )

    check_paths(df_train, "Train")
    check_paths(df_val, "Validation")
    check_paths(df_test, "Test")

    # 3. Verify Stratification
    print("\nVerifying stratification...")
    train_dist = df_train["breed"].value_counts(normalize=True).sort_index()
    val_dist = df_val["breed"].value_counts(normalize=True).sort_index()

    # Calculate absolute difference in proportions
    # Since we align by index (breed name), we can subtract directly.
    # We handle cases where a breed might be missing in one set (though stratification implies presence)
    all_breeds = sorted(list(set(train_dist.index) | set(val_dist.index)))
    train_dist = train_dist.reindex(all_breeds, fill_value=0)
    val_dist = val_dist.reindex(all_breeds, fill_value=0)

    diff = (train_dist - val_dist).abs()
    max_diff = diff.max()
    mean_diff = diff.mean()

    print(f"Max difference in class proportions: {max_diff:.6f}")
    print(f"Mean difference in class proportions: {mean_diff:.6f}")

    # Assert that the split is roughly stratified.
    # With 120 classes and ~9200 samples, avg samples per class is ~76.
    # 20% val is ~15 samples. 1/76 is approx 0.013.
    # We expect the distribution to be very close.
    # A loose threshold of 0.02 (2%) difference in any single class proportion is reasonable
    # given the small sample size per class.

    if max_diff > 0.05:  # 5% tolerance to be safe against very small classes
        raise AssertionError(
            "Stratification check failed: Class distributions differ significantly."
        )

    # Ensure all classes in train are in val (if sample size allows)
    # With stratification, this should hold unless a class has < 2 samples (checked earlier)
    train_classes = set(df_train["breed"].unique())
    val_classes = set(df_val["breed"].unique())

    if not val_classes.issubset(train_classes):
        raise AssertionError(
            "Validation set contains classes not present in Training set."
        )

    print("Stratification verification passed.")
    print("\nAll checks passed successfully.")


if __name__ == "__main__":
    try:
        train_p, val_p, test_p = generate_metadata()
        validate_metadata(train_p, val_p, test_p)
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        exit(1)
