import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


def main():
    # Configuration
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    RANDOM_STATE = 42
    VAL_SIZE = 0.2

    # Ensure metadata directory exists
    os.makedirs(METADATA_DIR, exist_ok=True)

    print("Loading raw data...")
    train_df = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(INPUT_DIR, "test.csv"))

    # --- Preprocessing Train Data ---
    # train.csv contains one row per StudyInstanceUID.
    # We add the relative path to the image directory.
    train_df["image_path"] = train_df["StudyInstanceUID"].apply(
        lambda x: os.path.join("train_images", x)
    )

    # --- Splitting Train/Val ---
    # We use StratifiedShuffleSplit based on 'patient_overall' to ensure
    # the fracture prevalence is balanced.
    # Since StudyInstanceUID is unique per row in train.csv, we don't strictly need GroupShuffleSplit,
    # but Stratified ensures we get enough positive cases in validation.

    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=VAL_SIZE, random_state=RANDOM_STATE
    )

    # Get the indices
    train_idx, val_idx = next(splitter.split(train_df, train_df["patient_overall"]))

    train_meta = train_df.iloc[train_idx].copy()
    val_meta = train_df.iloc[val_idx].copy()

    # --- Preprocessing Test Data ---
    # test.csv has rows for predictions (multiple rows per study).
    # We want a metadata file that lists unique studies for data loading.
    test_unique_studies = (
        test_df[["StudyInstanceUID"]].drop_duplicates().reset_index(drop=True)
    )
    test_unique_studies["image_path"] = test_unique_studies["StudyInstanceUID"].apply(
        lambda x: os.path.join("test_images", x)
    )
    test_meta = test_unique_studies

    # --- Saving Metadata ---
    print(f"Saving metadata to {METADATA_DIR}...")
    train_meta.to_csv(os.path.join(METADATA_DIR, "train_metadata.csv"), index=False)
    val_meta.to_csv(os.path.join(METADATA_DIR, "val_metadata.csv"), index=False)
    test_meta.to_csv(os.path.join(METADATA_DIR, "test_metadata.csv"), index=False)

    # --- Verification & Statistics ---
    print("\n=== Dataset Statistics ===")

    print(f"Train set shape: {train_meta.shape}")
    print(f"Validation set shape: {val_meta.shape}")
    print(f"Test set (unique studies) shape: {test_meta.shape}")

    # Class distribution
    print("\nTrain 'patient_overall' distribution:")
    print(train_meta["patient_overall"].value_counts(normalize=True))
    print("\nValidation 'patient_overall' distribution:")
    print(val_meta["patient_overall"].value_counts(normalize=True))

    # Verify Split Requirements
    print("\n=== Verifying Split ===")
    train_ids = set(train_meta["StudyInstanceUID"])
    val_ids = set(val_meta["StudyInstanceUID"])

    # Check for overlap
    overlap = train_ids.intersection(val_ids)
    if overlap:
        raise AssertionError(
            f"Found {len(overlap)} overlapping StudyInstanceUIDs between train and val."
        )
    print("No overlap between train and validation sets.")

    # Check ratio
    total_train_val = len(train_meta) + len(val_meta)
    actual_val_ratio = len(val_meta) / total_train_val
    print(f"Actual validation ratio: {actual_val_ratio:.4f} (Target: {VAL_SIZE})")

    # Verify File Paths
    print("\n=== Verifying File Paths ===")

    def check_paths(df, name):
        if df.empty:
            print(f"Skipping check for {name} (empty dataframe).")
            return

        sample_size = min(1000, len(df))
        sample = df.sample(n=sample_size, random_state=RANDOM_STATE)

        missing_count = 0
        missing_samples = []

        for _, row in sample.iterrows():
            rel_path = row["image_path"]
            # Resolve relative to INPUT_DIR
            full_path = os.path.join(INPUT_DIR, rel_path)

            # Check if it exists (it's a directory in this dataset)
            if not os.path.exists(full_path):
                missing_count += 1
                if len(missing_samples) < 5:
                    missing_samples.append(rel_path)

        missing_ratio = missing_count / sample_size
        print(
            f"[{name}] Checked {sample_size} paths. Missing ratio: {missing_ratio:.4f}"
        )

        if missing_ratio > 0.5:
            print("Sample missing paths:")
            for p in missing_samples:
                print(f"  {p}")
            raise FileNotFoundError(
                f"More than 50% of file paths in {name} are missing."
            )

    check_paths(train_meta, "Train Metadata")
    check_paths(val_meta, "Validation Metadata")
    check_paths(test_meta, "Test Metadata")

    print("\nMetadata generation and verification completed successfully.")


if __name__ == "__main__":
    main()
