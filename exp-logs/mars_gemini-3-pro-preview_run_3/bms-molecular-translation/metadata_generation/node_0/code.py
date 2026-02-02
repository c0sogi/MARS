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
    """Generates metadata CSVs for train, val, and test sets."""
    # Ensure metadata directory exists
    os.makedirs(METADATA_DIR, exist_ok=True)

    # --- 1. Process Training Data ---
    print("Loading train_labels.csv...")
    train_labels_path = os.path.join(INPUT_DIR, "train_labels.csv")
    if not os.path.exists(train_labels_path):
        raise FileNotFoundError(f"Could not find {train_labels_path}")

    df = pd.read_csv(train_labels_path)

    # Ensure image_id is string
    df["image_id"] = df["image_id"].astype(str)

    print("Generating training file paths...")
    # The dataset description states images are in a 3-level folder structure by image_id
    # e.g., image_id "000011a64c74" -> "train/0/0/0/000011a64c74.png"
    # We construct the path relative to the ./input directory
    c0 = df["image_id"].str[0]
    c1 = df["image_id"].str[1]
    c2 = df["image_id"].str[2]

    df["file_path"] = (
        "train/" + c0 + "/" + c1 + "/" + c2 + "/" + df["image_id"] + ".png"
    )

    # --- Stratified Split ---
    print("Performing stratified split...")
    # We use the length of the InChI string as a proxy for label complexity/distribution
    # since InChI strings are unique identifiers.
    df["inchi_len"] = df["InChI"].str.len()

    # Create bins for stratification (20 quantiles)
    # duplicates='drop' handles cases where bin edges might be identical
    df["stratify_bin"] = pd.qcut(df["inchi_len"], q=20, labels=False, duplicates="drop")

    train_df, val_df = train_test_split(
        df, test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=df["stratify_bin"]
    )

    # Clean up temporary columns
    train_df = train_df.drop(columns=["inchi_len", "stratify_bin"])
    val_df = val_df.drop(columns=["inchi_len", "stratify_bin"])

    print(f"Saving metadata to {METADATA_DIR}...")
    train_df.to_csv(os.path.join(METADATA_DIR, "train_metadata.csv"), index=False)
    val_df.to_csv(os.path.join(METADATA_DIR, "val_metadata.csv"), index=False)

    # --- 2. Process Test Data ---
    print("Loading sample_submission.csv for test data...")
    test_sample_path = os.path.join(INPUT_DIR, "sample_submission.csv")
    if not os.path.exists(test_sample_path):
        raise FileNotFoundError(f"Could not find {test_sample_path}")

    test_df = pd.read_csv(test_sample_path)
    test_df["image_id"] = test_df["image_id"].astype(str)

    print("Generating test file paths...")
    t0 = test_df["image_id"].str[0]
    t1 = test_df["image_id"].str[1]
    t2 = test_df["image_id"].str[2]

    test_df["file_path"] = (
        "test/" + t0 + "/" + t1 + "/" + t2 + "/" + test_df["image_id"] + ".png"
    )

    # Save test metadata
    test_df.to_csv(os.path.join(METADATA_DIR, "test_metadata.csv"), index=False)

    print("Metadata generation complete.")


def validate_generated_metadata():
    """Performs validation checks on the generated metadata files."""
    print("\n--- Validating Metadata ---")

    # Load the newly created files
    train_df = pd.read_csv(os.path.join(METADATA_DIR, "train_metadata.csv"))
    val_df = pd.read_csv(os.path.join(METADATA_DIR, "val_metadata.csv"))
    test_df = pd.read_csv(os.path.join(METADATA_DIR, "test_metadata.csv"))

    # 1. Summary Statistics
    print(f"Train set: {len(train_df)} samples")
    print(f"Val set:   {len(val_df)} samples")
    print(f"Test set:  {len(test_df)} samples")

    # 2. Check File Existence (Random Sample)
    def check_paths(df, name):
        sample_size = min(1000, len(df))
        sample = df.sample(n=sample_size, random_state=RANDOM_STATE)
        missing_paths = []

        for _, row in sample.iterrows():
            # Construct full path: ./input/ + relative_path_from_metadata
            full_path = os.path.join(INPUT_DIR, row["file_path"])
            if not os.path.exists(full_path):
                missing_paths.append(full_path)

        missing_ratio = len(missing_paths) / sample_size
        print(f"Missing file ratio for {name}: {missing_ratio:.4f}")

        if missing_ratio > 0.5:
            print("Examples of missing paths:")
            for p in missing_paths[:5]:
                print(f"  {p}")
            raise FileNotFoundError(
                f"High missing file ratio detected in {name} metadata!"
            )

    check_paths(train_df, "training")
    check_paths(val_df, "validation")
    check_paths(test_df, "test")

    # 3. Verify Stratification
    # We check if the distribution of InChI lengths is similar between train and val
    print("Verifying stratification consistency...")
    train_lens = train_df["InChI"].str.len()
    val_lens = val_df["InChI"].str.len()

    train_mean, train_std = train_lens.mean(), train_lens.std()
    val_mean, val_std = val_lens.mean(), val_lens.std()

    print(f"Train InChI length: mean={train_mean:.2f}, std={train_std:.2f}")
    print(f"Val InChI length:   mean={val_mean:.2f}, std={val_std:.2f}")

    # Allow for small deviation
    if abs(train_mean - val_mean) > 1.0 or abs(train_std - val_std) > 1.0:
        raise AssertionError(
            "Stratification check failed: Distribution of InChI lengths differs significantly between Train and Val."
        )

    print("All validation checks passed.")


if __name__ == "__main__":
    generate_metadata()
    validate_generated_metadata()
