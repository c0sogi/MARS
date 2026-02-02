import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Constants
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
RANDOM_STATE = 42
TRAIN_VAL_SPLIT_RATIO = 0.2


def generate_metadata():
    print("Starting metadata generation...")

    # Ensure metadata directory exists
    os.makedirs(METADATA_DIR, exist_ok=True)

    # Load raw csv files
    train_csv_path = os.path.join(INPUT_DIR, "train.csv")
    test_csv_path = os.path.join(INPUT_DIR, "test.csv")

    if not os.path.exists(train_csv_path):
        raise FileNotFoundError(f"Could not find {train_csv_path}")
    if not os.path.exists(test_csv_path):
        raise FileNotFoundError(f"Could not find {test_csv_path}")

    df_full_train = pd.read_csv(train_csv_path)
    df_test = pd.read_csv(test_csv_path)

    # Construct relative file paths
    # Based on dataset info, images are in train_images/ and test_images/ with .png extension
    df_full_train["file_path"] = (
        "train_images/" + df_full_train["id_code"].astype(str) + ".png"
    )
    df_test["file_path"] = "test_images/" + df_test["id_code"].astype(str) + ".png"

    # Perform Stratified Split
    print(f"Splitting training data with ratio {TRAIN_VAL_SPLIT_RATIO} (Stratified)...")
    df_train, df_val = train_test_split(
        df_full_train,
        test_size=TRAIN_VAL_SPLIT_RATIO,
        random_state=RANDOM_STATE,
        stratify=df_full_train["diagnosis"],
    )

    # Save metadata
    train_meta_path = os.path.join(METADATA_DIR, "train.csv")
    val_meta_path = os.path.join(METADATA_DIR, "val.csv")
    test_meta_path = os.path.join(METADATA_DIR, "test.csv")

    df_train.to_csv(train_meta_path, index=False)
    df_val.to_csv(val_meta_path, index=False)
    df_test.to_csv(test_meta_path, index=False)

    print("Metadata generation complete.")
    return train_meta_path, val_meta_path, test_meta_path


def check_file_paths(df, name):
    """
    Checks if files exist for a random sample of the dataframe.
    """
    sample_size = 1000
    if len(df) < sample_size:
        sample = df
    else:
        sample = df.sample(n=sample_size, random_state=RANDOM_STATE)

    missing_count = 0
    missing_samples = []

    for _, row in sample.iterrows():
        # Path in metadata is relative to ./input
        full_path = os.path.join(INPUT_DIR, row["file_path"])
        if not os.path.exists(full_path):
            missing_count += 1
            if len(missing_samples) < 5:
                missing_samples.append(row["file_path"])

    missing_ratio = missing_count / len(sample)
    print(
        f"[{name}] Missing file ratio: {missing_ratio:.4f} ({missing_count}/{len(sample)})"
    )

    if missing_ratio > 0.5:
        print(f"Sample missing paths: {missing_samples}")
        raise FileNotFoundError(
            f"More than 50% of files are missing in {name} dataset."
        )


def verify_stratification(df_train, df_val, target_col="diagnosis"):
    """
    Verifies that the split is stratified.
    """
    train_dist = df_train[target_col].value_counts(normalize=True).sort_index()
    val_dist = df_val[target_col].value_counts(normalize=True).sort_index()

    print("\nClass Distribution (Train):")
    print(train_dist)
    print("\nClass Distribution (Validation):")
    print(val_dist)

    # Check if distributions are reasonably close (within 1% absolute difference per class)
    diff = (train_dist - val_dist).abs()
    max_diff = diff.max()

    print(f"\nMaximum class distribution difference: {max_diff:.4f}")

    if max_diff > 0.015:  # Allow small variance due to discrete counts
        raise AssertionError(
            "Stratified split verification failed. Distributions differ significantly."
        )
    else:
        print("Stratification verification passed.")


def validate_metadata(train_path, val_path, test_path):
    print("\nStarting validation of generated metadata...")

    # Load datasets
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    # 1. Summary Statistics
    print("\n=== Summary Statistics ===")
    print(f"Train set size: {len(df_train)}")
    print(f"Validation set size: {len(df_val)}")
    print(f"Test set size: {len(df_test)}")

    print("\nTrain Diagnosis Counts:")
    print(df_train["diagnosis"].value_counts())

    # 2. Check File Paths
    print("\n=== Checking File Paths ===")
    check_file_paths(df_train, "Train")
    check_file_paths(df_val, "Validation")
    check_file_paths(df_test, "Test")

    # 3. Verify Stratification
    print("\n=== Verifying Stratification ===")
    verify_stratification(df_train, df_val)

    print("\nAll validation checks passed successfully.")


if __name__ == "__main__":
    try:
        # Generate
        t_path, v_path, te_path = generate_metadata()

        # Validate
        validate_metadata(t_path, v_path, te_path)

    except Exception as e:
        print(f"\nERROR: {e}")
        exit(1)
