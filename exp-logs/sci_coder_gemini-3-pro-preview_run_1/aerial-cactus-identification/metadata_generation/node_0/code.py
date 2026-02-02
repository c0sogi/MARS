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

    # Ensure metadata directory exists
    os.makedirs(METADATA_DIR, exist_ok=True)

    # Load raw data
    train_csv_path = os.path.join(INPUT_DIR, "train.csv")
    test_csv_path = os.path.join(INPUT_DIR, "sample_submission.csv")

    if not os.path.exists(train_csv_path):
        raise FileNotFoundError(f"Could not find {train_csv_path}")
    if not os.path.exists(test_csv_path):
        raise FileNotFoundError(f"Could not find {test_csv_path}")

    df_train_full = pd.read_csv(train_csv_path)
    df_test = pd.read_csv(test_csv_path)

    # Construct relative file paths
    # Train images are in 'train/' directory
    df_train_full["file_path"] = df_train_full["id"].apply(
        lambda x: os.path.join("train", x)
    )

    # Test images are in 'test/' directory
    df_test["file_path"] = df_test["id"].apply(lambda x: os.path.join("test", x))

    # Perform Stratified Split
    print(
        f"Splitting training data (Size: {len(df_train_full)}) into Train/Val with ratio {1-VAL_SIZE}:{VAL_SIZE}"
    )

    train_df, val_df = train_test_split(
        df_train_full,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=df_train_full["has_cactus"],
    )

    # Save metadata
    train_meta_path = os.path.join(METADATA_DIR, "train_metadata.csv")
    val_meta_path = os.path.join(METADATA_DIR, "val_metadata.csv")
    test_meta_path = os.path.join(METADATA_DIR, "test_metadata.csv")

    train_df.to_csv(train_meta_path, index=False)
    val_df.to_csv(val_meta_path, index=False)
    df_test.to_csv(test_meta_path, index=False)

    print("Metadata generation complete.")
    return train_meta_path, val_meta_path, test_meta_path


def validate_metadata(train_path, val_path, test_path):
    print("\nStarting metadata validation...")

    # Load metadata
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    # 1. Summary Statistics
    print("\nSummary Statistics:")
    print(f"Train set size: {len(df_train)}")
    print(f"Val set size:   {len(df_val)}")
    print(f"Test set size:  {len(df_test)}")

    train_dist = df_train["has_cactus"].value_counts(normalize=True).to_dict()
    val_dist = df_val["has_cactus"].value_counts(normalize=True).to_dict()

    print(f"Train Class Distribution: {train_dist}")
    print(f"Val Class Distribution:   {val_dist}")

    # 2. Check Stratification
    # We expect the distribution of class 1 to be very close
    train_ratio = df_train["has_cactus"].mean()
    val_ratio = df_val["has_cactus"].mean()
    diff = abs(train_ratio - val_ratio)

    print(
        f"Class 1 Ratio - Train: {train_ratio:.4f}, Val: {val_ratio:.4f}, Diff: {diff:.4f}"
    )

    if diff > 0.01:  # Allow small tolerance
        raise AssertionError(
            f"Stratification failed. Difference in class distribution ({diff}) is too large."
        )
    print("Stratification check passed.")

    # 3. Check File Paths
    def check_paths(df, name):
        print(f"Checking file paths for {name}...")
        # Sample 1000 paths or all if less than 1000
        n_sample = min(1000, len(df))
        sample_paths = (
            df["file_path"].sample(n=n_sample, random_state=RANDOM_STATE).values
        )

        missing_count = 0
        missing_samples = []

        for rel_path in sample_paths:
            # Path in metadata is relative to ./input
            full_path = os.path.join(INPUT_DIR, rel_path)
            if not os.path.exists(full_path):
                missing_count += 1
                if len(missing_samples) < 5:
                    missing_samples.append(rel_path)

        missing_ratio = missing_count / n_sample
        print(
            f"  Missing files: {missing_count}/{n_sample} (Ratio: {missing_ratio:.4f})"
        )

        if missing_ratio > 0.5:
            print("  Sample missing paths:", missing_samples)
            raise FileNotFoundError(
                f"More than 50% of file paths in {name} are missing."
            )

    check_paths(df_train, "Train")
    check_paths(df_val, "Validation")
    check_paths(df_test, "Test")

    print("\nAll validation checks passed successfully.")


if __name__ == "__main__":
    try:
        t_path, v_path, te_path = generate_metadata()
        validate_metadata(t_path, v_path, te_path)
    except Exception as e:
        print(f"\nERROR: {e}")
        exit(1)
