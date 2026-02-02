import pandas as pd
import numpy as np
import os
import glob
from sklearn.model_selection import StratifiedGroupKFold

# Configuration
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
TRAIN_CSV = os.path.join(INPUT_DIR, "train.csv")
SAMPLE_SUB = os.path.join(INPUT_DIR, "sample_submission.csv")
RANDOM_STATE = 42


def generate_metadata():
    # Ensure metadata directory exists
    os.makedirs(METADATA_DIR, exist_ok=True)

    # --- Process Training Data ---
    print("Loading training data...")
    df_train_full = pd.read_csv(TRAIN_CSV)

    # Construct relative file paths
    # Format: train/{image_id}.dicom
    df_train_full["file_path"] = df_train_full["image_id"].apply(
        lambda x: os.path.join("train", f"{x}.dicom")
    )

    # Split into Train and Validation
    # We use StratifiedGroupKFold to ensure:
    # 1. Groups (Images) are not split across sets (prevent leakage)
    # 2. Class distribution is preserved (Stratification)
    # 3. 80:20 split (5 folds, take one as validation)
    print("Splitting data into Train/Validation (80:20)...")
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # X is just a placeholder, y is the target for stratification, groups is for grouping
    split_generator = sgkf.split(
        X=df_train_full, y=df_train_full["class_id"], groups=df_train_full["image_id"]
    )

    # Take the first fold
    train_idx, val_idx = next(split_generator)

    df_train = df_train_full.iloc[train_idx].copy()
    df_val = df_train_full.iloc[val_idx].copy()

    # Save to metadata
    train_save_path = os.path.join(METADATA_DIR, "train_meta.csv")
    val_save_path = os.path.join(METADATA_DIR, "val_meta.csv")

    df_train.to_csv(train_save_path, index=False)
    df_val.to_csv(val_save_path, index=False)
    print(f"Saved train metadata to {train_save_path}")
    print(f"Saved validation metadata to {val_save_path}")

    # --- Process Test Data ---
    print("Loading test data...")
    df_test = pd.read_csv(SAMPLE_SUB)

    # Construct relative file paths
    # Format: test/{image_id}.dicom
    df_test["file_path"] = df_test["image_id"].apply(
        lambda x: os.path.join("test", f"{x}.dicom")
    )

    test_save_path = os.path.join(METADATA_DIR, "test_meta.csv")
    df_test.to_csv(test_save_path, index=False)
    print(f"Saved test metadata to {test_save_path}")

    return df_train, df_val, df_test


def check_file_existence(df, name):
    """Checks if files exist for a sample of paths."""
    print(f"Checking file existence for {name}...")
    n_samples = min(1000, len(df))
    sample_df = df.sample(n=n_samples, random_state=RANDOM_STATE)

    missing_paths = []
    for _, row in sample_df.iterrows():
        # Path in metadata is relative to ./input
        full_path = os.path.join(INPUT_DIR, row["file_path"])
        if not os.path.exists(full_path):
            missing_paths.append(full_path)

    missing_ratio = len(missing_paths) / n_samples
    print(f"  Missing file ratio: {missing_ratio:.4f}")

    if missing_ratio > 0.5:
        print("  Sample of missing paths:")
        for p in missing_paths[:5]:
            print(f"    {p}")
        raise FileNotFoundError(
            f"Missing file ratio for {name} is {missing_ratio}, which exceeds the 0.5 threshold."
        )


def validate_split(train_df, val_df):
    """Verifies the integrity of the train/val split."""
    print("Verifying split integrity...")

    train_ids = set(train_df["image_id"].unique())
    val_ids = set(val_df["image_id"].unique())

    # 1. Check for Overlap
    overlap = train_ids.intersection(val_ids)
    assert (
        len(overlap) == 0
    ), f"Data Leakage Detected! {len(overlap)} images are in both train and validation sets."

    # 2. Check Split Ratio
    n_train = len(train_ids)
    n_val = len(val_ids)
    total = n_train + n_val
    val_ratio = n_val / total

    print(f"  Train images: {n_train}")
    print(f"  Val images: {n_val}")
    print(f"  Validation Ratio: {val_ratio:.4f}")

    # Allow small deviation due to group sizes
    assert (
        0.15 <= val_ratio <= 0.25
    ), f"Validation split ratio {val_ratio:.4f} is outside acceptable range (0.15-0.25)."
    print("  Split verification passed.")


def print_stats(df, name):
    print(f"\n--- {name} Statistics ---")
    print(f"Total Rows: {len(df)}")
    print(f"Unique Images: {df['image_id'].nunique()}")
    if "class_name" in df.columns:
        print("Top 5 Classes:")
        print(df["class_name"].value_counts(normalize=True).head())


def main():
    # Generate Metadata
    train_df, val_df, test_df = generate_metadata()

    # Reload from disk to ensure integrity of saved files
    train_df = pd.read_csv(os.path.join(METADATA_DIR, "train_meta.csv"))
    val_df = pd.read_csv(os.path.join(METADATA_DIR, "val_meta.csv"))
    test_df = pd.read_csv(os.path.join(METADATA_DIR, "test_meta.csv"))

    # Print Statistics
    print_stats(train_df, "Training Set")
    print_stats(val_df, "Validation Set")
    print_stats(test_df, "Test Set")

    # Run Validations
    check_file_existence(train_df, "Training Set")
    check_file_existence(val_df, "Validation Set")
    check_file_existence(test_df, "Test Set")

    validate_split(train_df, val_df)

    print("\nMetadata generation and validation completed successfully.")


if __name__ == "__main__":
    main()
