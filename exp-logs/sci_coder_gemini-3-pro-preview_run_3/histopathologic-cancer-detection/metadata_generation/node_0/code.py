import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def main():
    # Configuration
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    TRAIN_LABELS_FILE = "train_labels.csv"
    SAMPLE_SUBMISSION_FILE = "sample_submission.csv"
    RANDOM_STATE = 42
    VAL_SIZE = 0.2

    # Ensure metadata directory exists
    if not os.path.exists(METADATA_DIR):
        os.makedirs(METADATA_DIR)
        print(f"Created directory: {METADATA_DIR}")

    print("Loading raw data...")
    # Load training labels
    train_labels_path = os.path.join(INPUT_DIR, TRAIN_LABELS_FILE)
    if not os.path.exists(train_labels_path):
        raise FileNotFoundError(f"Could not find {train_labels_path}")

    df_full = pd.read_csv(train_labels_path)

    # Load sample submission to get test IDs
    test_submission_path = os.path.join(INPUT_DIR, SAMPLE_SUBMISSION_FILE)
    if not os.path.exists(test_submission_path):
        raise FileNotFoundError(f"Could not find {test_submission_path}")

    df_test = pd.read_csv(test_submission_path)

    # Construct relative file paths
    # The dataset description indicates files are .tif and located in train/ and test/ folders
    df_full["file_path"] = df_full["id"].apply(
        lambda x: os.path.join("train", f"{x}.tif")
    )
    df_test["file_path"] = df_test["id"].apply(
        lambda x: os.path.join("test", f"{x}.tif")
    )

    print(f"Total available training samples: {len(df_full)}")
    print(f"Total test samples: {len(df_test)}")

    # Split training data into train and validation
    print(
        f"Splitting data with validation size {VAL_SIZE} and random state {RANDOM_STATE}..."
    )

    # Using stratified sampling on 'label' to ensure distribution consistency
    # No patient ID or group information is provided, so we assume independence of samples
    df_train, df_val = train_test_split(
        df_full,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=df_full["label"],
    )

    # Save metadata
    print("Saving metadata to ./metadata/ ...")
    train_save_path = os.path.join(METADATA_DIR, "train.csv")
    val_save_path = os.path.join(METADATA_DIR, "val.csv")
    test_save_path = os.path.join(METADATA_DIR, "test.csv")

    df_train.to_csv(train_save_path, index=False)
    df_val.to_csv(val_save_path, index=False)
    df_test.to_csv(test_save_path, index=False)

    print("Metadata generation complete.")

    # --- Verification Steps ---
    print("\n--- Verifying Metadata ---")

    # Reload datasets to verify integrity of saved files
    df_train_loaded = pd.read_csv(train_save_path)
    df_val_loaded = pd.read_csv(val_save_path)
    df_test_loaded = pd.read_csv(test_save_path)

    datasets = {
        "Train": df_train_loaded,
        "Validation": df_val_loaded,
        "Test": df_test_loaded,
    }

    # 1. Summary Statistics
    for name, df in datasets.items():
        print(f"\nSummary for {name} dataset:")
        print(f"  Shape: {df.shape}")
        if "label" in df.columns:
            print(f"  Class Distribution:\n{df['label'].value_counts(normalize=True)}")
            print(f"  Class Counts:\n{df['label'].value_counts()}")
        print(f"  Unique IDs: {df['id'].nunique()}")

    # 2. File Path Verification
    print("\nChecking file paths...")
    for name, df in datasets.items():
        print(f"  Checking {name} dataset paths...")
        # Select 1000 random paths or all if less than 1000
        n_samples = min(1000, len(df))
        sample_paths = (
            df["file_path"].sample(n=n_samples, random_state=RANDOM_STATE).tolist()
        )

        missing_count = 0
        missing_samples = []

        for rel_path in sample_paths:
            # Check if file exists relative to input directory
            full_path = os.path.join(INPUT_DIR, rel_path)
            if not os.path.exists(full_path):
                missing_count += 1
                if len(missing_samples) < 5:
                    missing_samples.append(rel_path)

        missing_ratio = missing_count / n_samples
        print(
            f"    Missing file ratio: {missing_ratio:.4f} ({missing_count}/{n_samples})"
        )

        if missing_ratio > 0.5:
            print(f"    Sample missing paths: {missing_samples}")
            raise FileNotFoundError(
                f"More than 50% of files missing in {name} dataset metadata."
            )

    # 3. Validation Split Verification
    print("\nVerifying validation split requirements...")

    # Check split ratio
    total_train_val = len(df_train_loaded) + len(df_val_loaded)
    actual_val_ratio = len(df_val_loaded) / total_train_val
    print(f"  Actual validation ratio: {actual_val_ratio:.4f} (Target: {VAL_SIZE})")

    # Assert ratio is correct (within small tolerance)
    assert (
        abs(actual_val_ratio - VAL_SIZE) < 1e-3
    ), f"Validation split ratio mismatch. Expected {VAL_SIZE}, got {actual_val_ratio}"

    # Check stratification
    train_dist = df_train_loaded["label"].value_counts(normalize=True)
    val_dist = df_val_loaded["label"].value_counts(normalize=True)

    print("  Train Label Distribution:\n", train_dist)
    print("  Val Label Distribution:\n", val_dist)

    # Assert distributions are similar (within 1%)
    for label in train_dist.index:
        if label in val_dist:
            diff = abs(train_dist[label] - val_dist[label])
            assert (
                diff < 0.01
            ), f"Stratification failed. Label {label} distribution differs by {diff:.4f}"
        else:
            raise AssertionError(
                f"Label {label} present in train but missing in validation set."
            )

    print("\nAll checks passed successfully.")


if __name__ == "__main__":
    main()
