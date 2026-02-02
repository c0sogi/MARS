import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Constants
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
TRAIN_LABELS_FILE = os.path.join(INPUT_DIR, "train_labels.csv")
SAMPLE_SUBMISSION_FILE = os.path.join(INPUT_DIR, "sample_submission.csv")
RANDOM_STATE = 42
VAL_SIZE = 0.2


def generate_file_path(image_id, split):
    """
    Generates the relative file path based on image_id and split (train/test).
    Structure: {split}/{char0}/{char1}/{char2}/{image_id}.png
    """
    return f"{split}/{image_id[0]}/{image_id[1]}/{image_id[2]}/{image_id}.png"


def verify_dataset(df, name):
    """
    Performs verification checks on a loaded dataset dataframe.
    """
    print(f"--- Verifying {name} Dataset ---")
    print(f"Shape: {df.shape}")
    if "InChI" in df.columns:
        print(f"Unique labels: {df['InChI'].nunique()}")

    # Check for missing files using a random sample
    sample_size = min(1000, len(df))
    sample = df.sample(n=sample_size, random_state=RANDOM_STATE)
    missing_count = 0
    missing_examples = []

    for _, row in sample.iterrows():
        rel_path = row["file_path"]
        # Construct full path to check existence
        full_path = os.path.join(INPUT_DIR, rel_path)
        if not os.path.exists(full_path):
            missing_count += 1
            if len(missing_examples) < 5:
                missing_examples.append(full_path)

    missing_ratio = missing_count / sample_size
    print(f"Missing file ratio: {missing_ratio:.4f}")

    if missing_ratio > 0.5:
        print("Examples of missing file paths:")
        for path in missing_examples:
            print(path)
        raise FileNotFoundError(
            f"Verification failed: More than 50% of sampled files are missing in {name} dataset."
        )

    print(f"{name} verification passed.\n")


def main():
    print("Starting metadata generation script...")

    # Ensure metadata directory exists
    os.makedirs(METADATA_DIR, exist_ok=True)

    # 1. Load Data
    print(f"Loading data from {INPUT_DIR}...")
    if not os.path.exists(TRAIN_LABELS_FILE):
        raise FileNotFoundError(f"Could not find {TRAIN_LABELS_FILE}")
    if not os.path.exists(SAMPLE_SUBMISSION_FILE):
        raise FileNotFoundError(f"Could not find {SAMPLE_SUBMISSION_FILE}")

    df_train_full = pd.read_csv(TRAIN_LABELS_FILE)
    df_test = pd.read_csv(SAMPLE_SUBMISSION_FILE)

    # 2. Generate File Paths
    print("Generating relative file paths...")
    # Apply path generation logic
    df_train_full["file_path"] = df_train_full["image_id"].apply(
        lambda x: generate_file_path(x, "train")
    )
    df_test["file_path"] = df_test["image_id"].apply(
        lambda x: generate_file_path(x, "test")
    )

    # 3. Split Training Data
    print(f"Splitting training data (Validation size: {VAL_SIZE})...")
    # We use a random split because InChI labels are unique strings, making
    # traditional class-based stratification impossible.
    df_train, df_val = train_test_split(
        df_train_full, test_size=VAL_SIZE, random_state=RANDOM_STATE, shuffle=True
    )

    # 4. Save Metadata
    print(f"Saving metadata to {METADATA_DIR}...")
    train_path = os.path.join(METADATA_DIR, "train_metadata.csv")
    val_path = os.path.join(METADATA_DIR, "val_metadata.csv")
    test_path = os.path.join(METADATA_DIR, "test_metadata.csv")

    df_train.to_csv(train_path, index=False)
    df_val.to_csv(val_path, index=False)
    df_test.to_csv(test_path, index=False)

    # 5. Verification
    print("Loading generated metadata for verification...")
    df_train_loaded = pd.read_csv(train_path)
    df_val_loaded = pd.read_csv(val_path)
    df_test_loaded = pd.read_csv(test_path)

    # Verify Split Ratio
    total_train_samples = len(df_train_full)
    val_count = len(df_val_loaded)
    actual_ratio = val_count / total_train_samples
    print(f"Actual validation ratio: {actual_ratio:.4f}")

    # Allow a tiny margin for rounding errors
    if not (0.199 < actual_ratio < 0.201):
        raise AssertionError(
            f"Validation split verification failed. Expected ~{VAL_SIZE}, got {actual_ratio:.4f}"
        )

    # Verify Content and File Existence
    verify_dataset(df_train_loaded, "Training")
    verify_dataset(df_val_loaded, "Validation")
    verify_dataset(df_test_loaded, "Test")

    print("Success! Metadata generation and verification complete.")


if __name__ == "__main__":
    main()
