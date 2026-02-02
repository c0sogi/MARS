import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import glob

# Constants
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
RANDOM_STATE = 42
VAL_SIZE = 0.2


def generate_metadata():
    """
    Reads raw data, processes it, creates a validation split, and saves metadata files.
    """
    print("Starting metadata generation...")

    # Ensure metadata directory exists
    os.makedirs(METADATA_DIR, exist_ok=True)

    # Load raw csvs
    train_csv_path = os.path.join(INPUT_DIR, "train.csv")
    test_csv_path = os.path.join(INPUT_DIR, "test.csv")

    if not os.path.exists(train_csv_path):
        raise FileNotFoundError(f"Train CSV not found at {train_csv_path}")
    if not os.path.exists(test_csv_path):
        raise FileNotFoundError(f"Test CSV not found at {test_csv_path}")

    df_train_full = pd.read_csv(train_csv_path)
    df_test = pd.read_csv(test_csv_path)

    # Add relative file paths
    # Structure is input/train/{id}.jpg and input/test/{id}.jpg
    # Metadata paths should be relative to input/, so: "train/{id}.jpg"
    df_train_full["file_path"] = df_train_full["Id"].apply(
        lambda x: os.path.join("train", f"{x}.jpg")
    )
    df_test["file_path"] = df_test["Id"].apply(
        lambda x: os.path.join("test", f"{x}.jpg")
    )

    # Create validation split
    # Since this is a regression task, we stratify by binning the target 'Pawpularity'
    # This ensures the validation set has a similar distribution of scores as the training set
    num_bins = int(np.floor(1 + np.log2(len(df_train_full))))  # Sturges' rule
    df_train_full["pawpularity_bins"] = pd.cut(
        df_train_full["Pawpularity"], bins=num_bins, labels=False
    )

    train_df, val_df = train_test_split(
        df_train_full,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=df_train_full["pawpularity_bins"],
        shuffle=True,
    )

    # Drop the temporary bin column
    train_df = train_df.drop(columns=["pawpularity_bins"])
    val_df = val_df.drop(columns=["pawpularity_bins"])
    df_train_full = df_train_full.drop(
        columns=["pawpularity_bins"]
    )  # Clean up original just in case

    # Save to metadata directory
    train_save_path = os.path.join(METADATA_DIR, "train.csv")
    val_save_path = os.path.join(METADATA_DIR, "validation.csv")
    test_save_path = os.path.join(METADATA_DIR, "test.csv")

    train_df.to_csv(train_save_path, index=False)
    val_df.to_csv(val_save_path, index=False)
    df_test.to_csv(test_save_path, index=False)

    print(f"Metadata saved to {METADATA_DIR}")
    return train_save_path, val_save_path, test_save_path


def check_file_paths(df, name):
    """
    Checks if a random sample of file paths in the dataframe exist in the input directory.
    """
    print(f"Checking file paths for {name} dataset...")
    sample_size = min(1000, len(df))
    sample = df.sample(n=sample_size, random_state=RANDOM_STATE)

    missing_count = 0
    missing_samples = []

    for _, row in sample.iterrows():
        # Path in metadata is relative to ./input
        rel_path = row["file_path"]
        full_path = os.path.join(INPUT_DIR, rel_path)

        if not os.path.exists(full_path):
            missing_count += 1
            if len(missing_samples) < 5:
                missing_samples.append(full_path)

    missing_ratio = missing_count / sample_size
    print(f"  Missing file ratio: {missing_ratio:.4f} ({missing_count}/{sample_size})")

    if missing_ratio > 0.5:
        print("  Sample of missing paths:")
        for p in missing_samples:
            print(f"    {p}")
        raise FileNotFoundError(
            f"More than 50% of file paths are missing for {name} dataset."
        )


def verify_metadata(train_path, val_path, test_path):
    """
    Loads generated metadata and performs validation checks.
    """
    print("\nVerifying generated metadata...")

    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    # 1. Summary Statistics
    print("\nSummary Statistics:")
    print(f"Train set shape: {df_train.shape}")
    print(f"Validation set shape: {df_val.shape}")
    print(f"Test set shape: {df_test.shape}")

    print("\nTarget Distribution (Pawpularity):")
    print(
        f"Train Mean: {df_train['Pawpularity'].mean():.2f}, Std: {df_train['Pawpularity'].std():.2f}"
    )
    print(
        f"Val Mean:   {df_val['Pawpularity'].mean():.2f}, Std: {df_val['Pawpularity'].std():.2f}"
    )

    # 2. File Path Checks
    check_file_paths(df_train, "Train")
    check_file_paths(df_val, "Validation")
    check_file_paths(df_test, "Test")

    # 3. Validation Logic Checks
    print("\nVerifying split logic...")

    # Check split ratio
    total_train_val = len(df_train) + len(df_val)
    actual_val_ratio = len(df_val) / total_train_val
    print(f"Actual validation ratio: {actual_val_ratio:.4f} (Target: {VAL_SIZE})")

    # Allow small deviation due to rounding/binning constraints
    if not (0.19 < actual_val_ratio < 0.21):
        raise AssertionError(
            f"Validation split ratio {actual_val_ratio:.4f} is significantly different from expected {VAL_SIZE}"
        )

    # Check for data leakage (overlap)
    train_ids = set(df_train["Id"])
    val_ids = set(df_val["Id"])
    overlap = train_ids.intersection(val_ids)

    if overlap:
        raise AssertionError(
            f"Found {len(overlap)} IDs overlapping between train and validation sets."
        )

    # Check stratification effectiveness (comparing means)
    # A simple check: means should be reasonably close
    mean_diff = abs(df_train["Pawpularity"].mean() - df_val["Pawpularity"].mean())
    if (
        mean_diff > 5.0
    ):  # Arbitrary threshold, but means should be close for stratified split
        raise AssertionError(
            f"Means of train and val sets differ significantly ({mean_diff:.2f}), stratification might have failed."
        )

    print("All validation checks passed successfully.")


if __name__ == "__main__":
    try:
        train_p, val_p, test_p = generate_metadata()
        verify_metadata(train_p, val_p, test_p)
        print("\nScript completed successfully.")
    except Exception as e:
        print(f"\nERROR: {e}")
        raise e
