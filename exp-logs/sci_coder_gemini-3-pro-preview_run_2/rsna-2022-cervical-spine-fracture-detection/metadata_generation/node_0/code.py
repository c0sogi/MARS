import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

# Constants
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
TRAIN_IMAGES_DIR = "train_images"
TEST_IMAGES_DIR = "test_images"
RANDOM_STATE = 42
VAL_SIZE = 0.2


def generate_metadata():
    """Generates metadata files for train, val, and test sets."""
    if not os.path.exists(METADATA_DIR):
        os.makedirs(METADATA_DIR)

    # --- Process Training Data ---
    train_path = os.path.join(INPUT_DIR, "train.csv")
    train_df = pd.read_csv(train_path)

    # Add relative path to the study directory
    # Structure: ./input/train_images/[StudyInstanceUID]
    train_df["image_path"] = train_df["StudyInstanceUID"].apply(
        lambda x: os.path.join(TRAIN_IMAGES_DIR, x)
    )

    # Split into Train and Validation
    # We stratify by 'patient_overall' to ensure balanced fracture cases
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=VAL_SIZE, random_state=RANDOM_STATE
    )

    # Get indices
    train_idx, val_idx = next(splitter.split(train_df, train_df["patient_overall"]))

    train_split = train_df.iloc[train_idx].copy()
    val_split = train_df.iloc[val_idx].copy()

    # Save to metadata
    train_split.to_csv(os.path.join(METADATA_DIR, "train_metadata.csv"), index=False)
    val_split.to_csv(os.path.join(METADATA_DIR, "val_metadata.csv"), index=False)

    # --- Process Test Data ---
    test_path = os.path.join(INPUT_DIR, "test.csv")
    test_df = pd.read_csv(test_path)

    # test.csv has multiple rows per study (one for each prediction target)
    # We need a study-level metadata file
    test_studies = (
        test_df[["StudyInstanceUID"]].drop_duplicates().reset_index(drop=True)
    )

    # Add relative path
    test_studies["image_path"] = test_studies["StudyInstanceUID"].apply(
        lambda x: os.path.join(TEST_IMAGES_DIR, x)
    )

    # Save to metadata
    test_studies.to_csv(os.path.join(METADATA_DIR, "test_metadata.csv"), index=False)


def check_file_paths(df, dataset_name):
    """Checks if a random sample of file paths exist."""
    paths = df["image_path"].tolist()

    # Sample 1000 paths if dataset is large enough
    if len(paths) > 1000:
        sample_paths = np.random.choice(paths, 1000, replace=False)
    else:
        sample_paths = paths

    missing_count = 0
    missing_samples = []

    for rel_path in sample_paths:
        full_path = os.path.join(INPUT_DIR, rel_path)
        if not os.path.exists(full_path):
            missing_count += 1
            if len(missing_samples) < 5:
                missing_samples.append(rel_path)

    total = len(sample_paths)
    ratio = missing_count / total if total > 0 else 0

    print(f"[{dataset_name}] Checked {total} paths. Missing ratio: {ratio:.4f}")

    if ratio > 0.5:
        print(f"Sample missing paths: {missing_samples}")
        raise FileNotFoundError(
            f"More than 50% of files missing for {dataset_name} dataset."
        )


def validate_metadata():
    """Loads generated metadata and performs validation checks."""
    print("Loading metadata for validation...")
    train_meta = pd.read_csv(os.path.join(METADATA_DIR, "train_metadata.csv"))
    val_meta = pd.read_csv(os.path.join(METADATA_DIR, "val_metadata.csv"))
    test_meta = pd.read_csv(os.path.join(METADATA_DIR, "test_metadata.csv"))

    # 1. Summary Statistics
    print("\n=== Summary Statistics ===")
    print(f"Train Set: {train_meta.shape[0]} samples")
    print(
        f"Class Distribution (patient_overall):\n{train_meta['patient_overall'].value_counts(normalize=True)}"
    )

    print(f"\nValidation Set: {val_meta.shape[0]} samples")
    print(
        f"Class Distribution (patient_overall):\n{val_meta['patient_overall'].value_counts(normalize=True)}"
    )

    print(f"\nTest Set: {test_meta.shape[0]} unique studies")

    # 2. File Path Checks
    print("\n=== File Path Verification ===")
    check_file_paths(train_meta, "Train")
    check_file_paths(val_meta, "Validation")
    check_file_paths(test_meta, "Test")

    # 3. Split Verification
    print("\n=== Split Verification ===")

    # Check for overlap
    train_ids = set(train_meta["StudyInstanceUID"])
    val_ids = set(val_meta["StudyInstanceUID"])
    overlap = train_ids.intersection(val_ids)

    if overlap:
        raise AssertionError(
            f"Found {len(overlap)} overlapping StudyInstanceUIDs between train and validation sets."
        )
    else:
        print("No overlap between train and validation sets.")

    # Check Stratification
    train_mean = train_meta["patient_overall"].mean()
    val_mean = val_meta["patient_overall"].mean()
    diff = abs(train_mean - val_mean)

    print(f"Train Positive Rate: {train_mean:.4f}")
    print(f"Val Positive Rate:   {val_mean:.4f}")
    print(f"Difference:          {diff:.4f}")

    # Allow a small tolerance for stratification differences
    if diff > 0.05:
        raise AssertionError(
            "Stratification failed: Significant difference in class distribution between train and val."
        )

    print("\nAll validation checks passed successfully.")


if __name__ == "__main__":
    # Set random seed for numpy as well just in case
    np.random.seed(RANDOM_STATE)

    try:
        generate_metadata()
        validate_metadata()
    except Exception as e:
        print(f"\nERROR: Script failed with exception: {e}")
        raise e
