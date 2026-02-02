import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GroupShuffleSplit


def generate_metadata():
    # Define paths
    input_dir = "./input"
    metadata_dir = "./metadata"
    train_labels_path = os.path.join(input_dir, "train_labels.csv")
    sample_submission_path = os.path.join(input_dir, "sample_submission.csv")

    # Create metadata directory
    os.makedirs(metadata_dir, exist_ok=True)

    # ---------------------------------------------------------
    # 1. Process Training Data
    # ---------------------------------------------------------
    print(f"Loading training labels from {train_labels_path}...")
    df_train_full = pd.read_csv(train_labels_path)

    # Generate file paths relative to ./input
    # Structure: train/{char0}/{char1}/{char2}/{image_id}.png
    print("Generating training file paths...")
    image_ids = df_train_full["image_id"].astype(str)
    df_train_full["file_path"] = (
        "train/"
        + image_ids.str[0]
        + "/"
        + image_ids.str[1]
        + "/"
        + image_ids.str[2]
        + "/"
        + image_ids
        + ".png"
    )

    # Analyze data for split strategy
    n_samples = len(df_train_full)
    n_unique_labels = df_train_full["InChI"].nunique()

    print(f"Total training samples: {n_samples}")
    print(f"Unique InChI labels: {n_unique_labels}")

    # Constants
    RANDOM_STATE = 42
    TEST_SIZE = 0.2

    # Determine split strategy
    # If there are significantly fewer unique labels than samples, it implies duplicates/groups.
    # We use GroupShuffleSplit to strictly separate molecules.
    # Otherwise, we use random split.
    if n_unique_labels < n_samples:
        print(
            "Duplicate labels detected. Performing Group Sampling split based on InChI labels."
        )
        gss = GroupShuffleSplit(
            n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        # split returns indices
        train_idx, val_idx = next(
            gss.split(df_train_full, groups=df_train_full["InChI"])
        )
        df_train = df_train_full.iloc[train_idx].copy()
        df_val = df_train_full.iloc[val_idx].copy()
        split_type = "group"
    else:
        print("Labels appear unique (or near unique). Performing Random Shuffle split.")
        df_train, df_val = train_test_split(
            df_train_full, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True
        )
        split_type = "random"

    print(f"Train set size: {len(df_train)}")
    print(f"Validation set size: {len(df_val)}")

    # Save to metadata
    print("Saving training and validation metadata...")
    df_train.to_csv(os.path.join(metadata_dir, "train.csv"), index=False)
    df_val.to_csv(os.path.join(metadata_dir, "val.csv"), index=False)

    # ---------------------------------------------------------
    # 2. Process Test Data
    # ---------------------------------------------------------
    print(f"Loading sample submission from {sample_submission_path}...")
    df_test = pd.read_csv(sample_submission_path)

    print("Generating test file paths...")
    test_ids = df_test["image_id"].astype(str)
    df_test["file_path"] = (
        "test/"
        + test_ids.str[0]
        + "/"
        + test_ids.str[1]
        + "/"
        + test_ids.str[2]
        + "/"
        + test_ids
        + ".png"
    )

    # Keep only necessary columns for metadata
    # We keep image_id and file_path. InChI in sample_submission is usually a placeholder.
    df_test_metadata = df_test[["image_id", "file_path"]].copy()

    print("Saving test metadata...")
    df_test_metadata.to_csv(os.path.join(metadata_dir, "test.csv"), index=False)

    return split_type


def validate_metadata(split_type):
    print("\n" + "=" * 40)
    print("Starting Validation Checks")
    print("=" * 40)

    input_dir = "./input"
    metadata_dir = "./metadata"

    # Load datasets
    df_train = pd.read_csv(os.path.join(metadata_dir, "train.csv"))
    df_val = pd.read_csv(os.path.join(metadata_dir, "val.csv"))
    df_test = pd.read_csv(os.path.join(metadata_dir, "test.csv"))

    # 1. Print Summary Statistics
    print("\n[Summary Statistics]")
    print(f"Train set shape: {df_train.shape}")
    print(f"Val set shape:   {df_val.shape}")
    print(f"Test set shape:  {df_test.shape}")
    print(f"Train unique labels: {df_train['InChI'].nunique()}")
    print(f"Val unique labels:   {df_val['InChI'].nunique()}")

    # 2. Check File Existence
    print("\n[File Existence Check]")

    def check_files(df, name):
        if len(df) == 0:
            print(f"{name}: Empty dataset.")
            return

        sample_size = min(1000, len(df))
        sample = df.sample(n=sample_size, random_state=42)
        missing_count = 0
        missing_examples = []

        for _, row in sample.iterrows():
            # Path is relative to ./input
            rel_path = row["file_path"]
            full_path = os.path.join(input_dir, rel_path)

            if not os.path.exists(full_path):
                missing_count += 1
                if len(missing_examples) < 5:
                    missing_examples.append(full_path)

        ratio = missing_count / sample_size
        print(
            f"{name}: Missing file ratio = {ratio:.4f} ({missing_count}/{sample_size})"
        )

        if ratio > 0.5:
            print(f"CRITICAL ERROR: High missing file ratio in {name}.")
            print("Example missing paths:")
            for p in missing_examples:
                print(f"  - {p}")
            raise FileNotFoundError(
                f"More than 50% of files missing in {name} dataset."
            )

    check_files(df_train, "Train")
    check_files(df_val, "Validation")
    check_files(df_test, "Test")

    # 3. Verify Split Requirements
    print("\n[Split Verification]")

    # Check Ratio
    total_train_val = len(df_train) + len(df_val)
    train_ratio = len(df_train) / total_train_val
    print(f"Train ratio: {train_ratio:.4f} (Target: ~0.8)")

    # Assert ratio is within reasonable bounds (e.g., +/- 5%)
    assert (
        0.75 < train_ratio < 0.85
    ), f"Train ratio {train_ratio:.4f} deviates significantly from 0.8"

    # Check ID Overlap
    train_ids = set(df_train["image_id"])
    val_ids = set(df_val["image_id"])
    intersection = train_ids.intersection(val_ids)
    assert (
        len(intersection) == 0
    ), f"Data Leakage: Found {len(intersection)} overlapping image_ids between train and val."

    # Check Group Split Integrity (if applicable)
    if split_type == "group":
        print("Verifying group split integrity (no label overlap)...")
        train_labels = set(df_train["InChI"])
        val_labels = set(df_val["InChI"])
        label_intersection = train_labels.intersection(val_labels)
        assert (
            len(label_intersection) == 0
        ), f"Group Split Failed: Found {len(label_intersection)} overlapping labels."

    print("\nAll validation checks passed successfully.")


if __name__ == "__main__":
    split_type = generate_metadata()
    validate_metadata(split_type)
