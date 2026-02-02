import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
import sys

# Configuration
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
TRAIN_CSV = os.path.join(INPUT_DIR, "train.csv")
TEST_DIR = os.path.join(INPUT_DIR, "test")
TRAIN_IMG_DIR = os.path.join(INPUT_DIR, "train")
RANDOM_STATE = 42
VAL_SIZE = 0.2


def generate_metadata():
    print("Starting metadata generation...")
    os.makedirs(METADATA_DIR, exist_ok=True)

    # 1. Process Training Data
    print(f"Reading {TRAIN_CSV}...")
    # Read attribute_ids as string to handle space-separated lists
    df = pd.read_csv(TRAIN_CSV, dtype={"id": str, "attribute_ids": str})

    # Construct relative file paths
    # Assuming .png based on dataset description.
    # In a real scenario, we might check extensions, but here we assume consistency or fix it.
    # The provided file list shows .png.
    df["file_path"] = "train/" + df["id"] + ".png"

    # Split into Train and Validation
    # We use random split with a fixed seed. For 120k samples and multi-label data,
    # random splitting is generally sufficient to preserve distribution and is much faster
    # than iterative stratification which can be computationally prohibitive.
    print(f"Splitting data (Test size: {VAL_SIZE}, Random State: {RANDOM_STATE})...")
    train_df, val_df = train_test_split(
        df, test_size=VAL_SIZE, random_state=RANDOM_STATE, shuffle=True
    )

    # Save Train and Val metadata
    train_metadata_path = os.path.join(METADATA_DIR, "train.csv")
    val_metadata_path = os.path.join(METADATA_DIR, "val.csv")

    train_df.to_csv(train_metadata_path, index=False)
    val_df.to_csv(val_metadata_path, index=False)
    print(f"Saved {train_metadata_path} ({len(train_df)} samples)")
    print(f"Saved {val_metadata_path} ({len(val_df)} samples)")

    # 2. Process Test Data
    print("Processing test files...")
    # List all files in test directory
    test_files = glob.glob(os.path.join(TEST_DIR, "*"))
    test_data = []
    for filepath in test_files:
        filename = os.path.basename(filepath)
        file_id = os.path.splitext(filename)[0]
        # Path relative to ./input
        rel_path = os.path.join("test", filename)
        test_data.append({"id": file_id, "file_path": rel_path})

    test_df = pd.DataFrame(test_data)
    test_metadata_path = os.path.join(METADATA_DIR, "test.csv")
    test_df.to_csv(test_metadata_path, index=False)
    print(f"Saved {test_metadata_path} ({len(test_df)} samples)")

    return train_metadata_path, val_metadata_path, test_metadata_path


def verify_metadata(train_path, val_path, test_path):
    print("\nVerifying metadata...")

    # Load datasets
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    # 1. Summary Statistics
    print("-" * 30)
    print("Dataset Summary:")
    print(f"Train set: {train_df.shape[0]} samples")
    print(f"Val set:   {val_df.shape[0]} samples")
    print(f"Test set:  {test_df.shape[0]} samples")
    print("-" * 30)

    # 2. File Path Verification
    print("Checking file paths...")

    def check_paths(df, name):
        if "file_path" not in df.columns:
            return

        # Sample 1000 paths
        sample_paths = (
            df["file_path"]
            .sample(n=min(1000, len(df)), random_state=RANDOM_STATE)
            .tolist()
        )
        missing_count = 0
        missing_samples = []

        for p in sample_paths:
            # Resolve full path: ./input + relative_path
            full_path = os.path.join(INPUT_DIR, p)
            if not os.path.exists(full_path):
                missing_count += 1
                if len(missing_samples) < 5:
                    missing_samples.append(p)

        missing_ratio = missing_count / len(sample_paths)
        print(f"[{name}] Missing file ratio: {missing_ratio:.4f}")

        if missing_ratio > 0.5:
            print(f"Sample missing paths from {name}: {missing_samples}")
            raise FileNotFoundError(f"Too many missing files in {name} dataset!")

    check_paths(train_df, "Train")
    check_paths(val_df, "Val")
    check_paths(test_df, "Test")

    # 3. Validation Split Verification (Stratification Check)
    print("Verifying validation split stratification...")

    def get_label_counts(df):
        # Parse space-separated attribute_ids
        all_labels = []
        for ids in df["attribute_ids"].dropna():
            all_labels.extend(ids.split(" "))
        return Counter(all_labels)

    train_counts = get_label_counts(train_df)
    val_counts = get_label_counts(val_df)

    # Create a DataFrame to compare distributions
    all_keys = set(train_counts.keys()) | set(val_counts.keys())
    dist_data = []
    for k in all_keys:
        dist_data.append(
            {
                "label": k,
                "train_freq": train_counts.get(k, 0) / len(train_df),
                "val_freq": val_counts.get(k, 0) / len(val_df),
            }
        )

    dist_df = pd.DataFrame(dist_data)

    # Calculate correlation between label frequencies
    correlation = dist_df["train_freq"].corr(dist_df["val_freq"])
    print(f"Label frequency correlation between Train and Val: {correlation:.4f}")

    # Assert that the split is representative
    # A high correlation (> 0.9) indicates that the relative frequency of labels is preserved.
    if correlation < 0.9:
        raise AssertionError(
            "Validation set distribution does not match training set distribution significantly."
        )

    print("Verification passed successfully.")


if __name__ == "__main__":
    try:
        train_p, val_p, test_p = generate_metadata()
        verify_metadata(train_p, val_p, test_p)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
