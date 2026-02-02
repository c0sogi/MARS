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
    """
    Reads raw data, creates train/val splits, and saves metadata files.
    """
    # Create metadata directory if it doesn't exist
    os.makedirs(METADATA_DIR, exist_ok=True)

    # 1. Load Data
    print("Loading raw data...")
    train_path = os.path.join(INPUT_DIR, "train.csv")
    test_path = os.path.join(INPUT_DIR, "test.csv")

    # Read CSVs
    # Using 'on_bad_lines' to skip potential malformed lines if any, though dataset should be clean.
    df_train_full = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    # 2. Split Training Data (Stratified)
    print(f"Splitting training data (Val size: {VAL_SIZE}, Stratify: score)...")

    # Stratified split based on 'score'
    # We shuffle the data with a fixed random state for reproducibility
    train_df, val_df = train_test_split(
        df_train_full,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=df_train_full["score"],
    )

    # 3. Add Metadata Columns
    # We add a 'source_file' column representing the relative path to the original file
    # This satisfies the requirement to store relative file paths in metadata.
    train_df["source_file"] = "train.csv"
    val_df["source_file"] = "train.csv"
    df_test["source_file"] = "test.csv"

    # 4. Save Metadata
    print("Saving metadata to ./metadata...")
    train_meta_path = os.path.join(METADATA_DIR, "train_metadata.csv")
    val_meta_path = os.path.join(METADATA_DIR, "val_metadata.csv")
    test_meta_path = os.path.join(METADATA_DIR, "test_metadata.csv")

    train_df.to_csv(train_meta_path, index=False)
    val_df.to_csv(val_meta_path, index=False)
    df_test.to_csv(test_meta_path, index=False)

    print("Metadata generation complete.")
    return train_meta_path, val_meta_path, test_meta_path


def validate_metadata(train_path, val_path, test_path):
    """
    Loads generated metadata and performs validation checks.
    """
    print("\nStarting validation checks...")

    # Load datasets
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    # 1. Print Summary Statistics
    print("\n=== Summary Statistics ===")
    print(f"Train set shape: {df_train.shape}")
    print(f"Validation set shape: {df_val.shape}")
    print(f"Test set shape: {df_test.shape}")

    print("\nTrain Score Distribution:")
    print(df_train["score"].value_counts().sort_index())

    print("\nValidation Score Distribution:")
    print(df_val["score"].value_counts().sort_index())

    # 2. Check File Paths
    # We check the 'source_file' column which contains relative paths to ./input
    print("\nChecking file paths...")

    def check_paths(df, name):
        if "source_file" not in df.columns:
            return

        # Select up to 1000 random paths
        n_samples = min(1000, len(df))
        sample_paths = df["source_file"].sample(n=n_samples, random_state=RANDOM_STATE)

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
            print("Sample missing paths:", missing_samples)
            raise FileNotFoundError(
                f"More than 50% of file paths in {name} are invalid."
            )

    check_paths(df_train, "Train")
    check_paths(df_val, "Validation")
    check_paths(df_test, "Test")

    # 3. Verify Stratification
    print("\nVerifying stratification...")

    # Calculate normalized value counts
    train_dist = df_train["score"].value_counts(normalize=True).sort_index()
    val_dist = df_val["score"].value_counts(normalize=True).sort_index()

    print("Train distribution (ratios):\n", train_dist)
    print("Val distribution (ratios):\n", val_dist)

    # Check if distributions are similar (tolerance of 1%)
    # We expect them to be very close due to stratified split
    diff = (train_dist - val_dist).abs()
    max_diff = diff.max()

    print(f"Max difference in class proportions: {max_diff:.4f}")

    if max_diff > 0.015:  # Allow small margin for rounding/small classes
        raise AssertionError(
            f"Stratification failed! Max difference in class proportions is {max_diff:.4f}, expected < 0.015"
        )

    print("Stratification check passed.")
    print("\nAll checks passed successfully.")


if __name__ == "__main__":
    # Generate
    t_path, v_path, te_path = generate_metadata()

    # Validate
    validate_metadata(t_path, v_path, te_path)
