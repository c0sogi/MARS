import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Configuration Constants
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
RANDOM_STATE = 42
VAL_SIZE = 0.2


def main():
    print("Starting metadata generation process...")

    # Create metadata directory
    os.makedirs(METADATA_DIR, exist_ok=True)

    # =========================================================================
    # 1. Process Training and Validation Data
    # =========================================================================
    train_meta_path = os.path.join(INPUT_DIR, "train_meta.parquet")
    print(f"Loading training metadata from {train_meta_path}...")

    if not os.path.exists(train_meta_path):
        raise FileNotFoundError(f"Input file not found: {train_meta_path}")

    # Load raw training metadata
    train_df = pd.read_parquet(train_meta_path)

    # Generate relative file paths for the batch files
    # The files are located at input/train/batch_{batch_id}.parquet
    # We store the path relative to input/
    print("Generating batch file paths...")
    train_df["batch_file_path"] = (
        "train/batch_" + train_df["batch_id"].astype(str) + ".parquet"
    )

    # Split into Train and Validation sets
    # Strategy: Random shuffle split (80/20)
    print(
        f"Splitting data into Train (80%) and Validation (20%) with random_state={RANDOM_STATE}..."
    )
    train_split, val_split = train_test_split(
        train_df, test_size=VAL_SIZE, random_state=RANDOM_STATE, shuffle=True
    )

    # Save generated metadata
    train_out_path = os.path.join(METADATA_DIR, "train_metadata.parquet")
    val_out_path = os.path.join(METADATA_DIR, "val_metadata.parquet")

    print(f"Saving training metadata to {train_out_path}...")
    train_split.to_parquet(train_out_path, index=False)

    print(f"Saving validation metadata to {val_out_path}...")
    val_split.to_parquet(val_out_path, index=False)

    # Free memory
    del train_df, train_split, val_split

    # =========================================================================
    # 2. Process Test Data
    # =========================================================================
    test_meta_path = os.path.join(INPUT_DIR, "test_meta.parquet")
    print(f"Loading test metadata from {test_meta_path}...")

    if not os.path.exists(test_meta_path):
        raise FileNotFoundError(f"Input file not found: {test_meta_path}")

    test_df = pd.read_parquet(test_meta_path)

    # Generate relative file paths for test batches
    print("Generating test batch file paths...")
    test_df["batch_file_path"] = (
        "test/batch_" + test_df["batch_id"].astype(str) + ".parquet"
    )

    # Save generated metadata
    test_out_path = os.path.join(METADATA_DIR, "test_metadata.parquet")
    print(f"Saving test metadata to {test_out_path}...")
    test_df.to_parquet(test_out_path, index=False)

    del test_df

    # =========================================================================
    # 3. Verification and Checks
    # =========================================================================
    print("\n" + "=" * 40)
    print("Performing Verification Checks")
    print("=" * 40)

    # Reload the generated files
    train_meta = pd.read_parquet(train_out_path)
    val_meta = pd.read_parquet(val_out_path)
    test_meta = pd.read_parquet(test_out_path)

    # --- Summary Statistics ---
    print("\nDataset Summary:")
    print(f"Training Set Samples:   {len(train_meta)}")
    print(f"Validation Set Samples: {len(val_meta)}")
    print(f"Test Set Samples:       {len(test_meta)}")

    # --- Verify Split Ratio ---
    total_train_val = len(train_meta) + len(val_meta)
    actual_val_ratio = len(val_meta) / total_train_val
    print(f"Actual Validation Ratio: {actual_val_ratio:.5f} (Target: {VAL_SIZE})")

    if abs(actual_val_ratio - VAL_SIZE) > 0.01:
        raise AssertionError(
            f"Validation split ratio {actual_val_ratio} deviates significantly from target {VAL_SIZE}"
        )

    # --- Verify No Leakage ---
    print("Checking for data leakage (event_id overlap)...")
    train_ids = set(train_meta["event_id"])
    val_ids = set(val_meta["event_id"])
    intersection = train_ids.intersection(val_ids)

    if len(intersection) > 0:
        raise AssertionError(
            f"Data leakage detected! {len(intersection)} events found in both train and validation sets."
        )
    print("No leakage detected.")

    # --- Verify File Paths ---
    def check_paths(df, name):
        print(f"\nChecking file paths for {name}...")
        # Sample 1000 paths
        sample_size = 1000
        if len(df) > sample_size:
            sample = df.sample(n=sample_size, random_state=RANDOM_STATE)
        else:
            sample = df

        missing_count = 0
        missing_samples = []

        for _, row in sample.iterrows():
            rel_path = row["batch_file_path"]
            full_path = os.path.join(INPUT_DIR, rel_path)

            if not os.path.exists(full_path):
                missing_count += 1
                if len(missing_samples) < 5:
                    missing_samples.append(rel_path)

        missing_ratio = missing_count / len(sample)
        print(f"Missing file ratio: {missing_ratio:.4f}")

        if missing_ratio > 0.5:
            print("Sample missing paths:", missing_samples)
            raise FileNotFoundError(
                f"More than 50% of file paths in {name} do not resolve to existing files."
            )

    check_paths(train_meta, "Training Set")
    check_paths(val_meta, "Validation Set")
    check_paths(test_meta, "Test Set")

    print("\nMetadata generation and verification completed successfully.")


if __name__ == "__main__":
    main()
