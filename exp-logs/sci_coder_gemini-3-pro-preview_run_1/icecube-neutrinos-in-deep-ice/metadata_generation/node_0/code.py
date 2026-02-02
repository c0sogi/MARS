import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def main():
    # Define directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Ensure metadata directory exists
    os.makedirs(METADATA_DIR, exist_ok=True)

    print("Starting metadata generation...")

    # ==========================================
    # 1. Process Training and Validation Data
    # ==========================================
    train_meta_path = os.path.join(INPUT_DIR, "train_meta.parquet")
    print(f"Loading {train_meta_path}...")

    # Load training metadata
    # This file contains batch_id, event_id, first/last_pulse_index, azimuth, zenith
    df_train_full = pd.read_parquet(train_meta_path)

    # Construct relative file paths based on batch_id
    # Format: train/batch_[id].parquet
    df_train_full["file_path"] = (
        "train/batch_" + df_train_full["batch_id"].astype(str) + ".parquet"
    )

    print(f"Total training events found: {len(df_train_full)}")

    # Split into Train (80%) and Validation (20%)
    # Using random_state=42 for reproducibility
    print("Splitting data into train (80%) and validation (20%)...")
    df_train, df_val = train_test_split(
        df_train_full, test_size=0.2, random_state=42, shuffle=True
    )

    # Save to metadata directory
    print("Saving train_metadata.parquet...")
    df_train.to_parquet(
        os.path.join(METADATA_DIR, "train_metadata.parquet"), index=False
    )

    print("Saving val_metadata.parquet...")
    df_val.to_parquet(os.path.join(METADATA_DIR, "val_metadata.parquet"), index=False)

    # Clean up memory
    del df_train_full, df_train, df_val

    # ==========================================
    # 2. Process Test Data
    # ==========================================
    test_meta_path = os.path.join(INPUT_DIR, "test_meta.parquet")
    print(f"Loading {test_meta_path}...")

    df_test = pd.read_parquet(test_meta_path)

    # Construct relative file paths for test set
    # Format: test/batch_[id].parquet
    df_test["file_path"] = "test/batch_" + df_test["batch_id"].astype(str) + ".parquet"

    print("Saving test_metadata.parquet...")
    df_test.to_parquet(os.path.join(METADATA_DIR, "test_metadata.parquet"), index=False)

    del df_test

    # ==========================================
    # 3. Verification and Checks
    # ==========================================
    print("\n" + "=" * 30)
    print("PERFORMING VALIDATION CHECKS")
    print("=" * 30)

    # Load generated metadata
    meta_train = pd.read_parquet(os.path.join(METADATA_DIR, "train_metadata.parquet"))
    meta_val = pd.read_parquet(os.path.join(METADATA_DIR, "val_metadata.parquet"))
    meta_test = pd.read_parquet(os.path.join(METADATA_DIR, "test_metadata.parquet"))

    # 3.1 Summary Statistics
    print("\n--- Summary Statistics ---")
    print(f"Train samples: {len(meta_train)}")
    print(f"Val samples:   {len(meta_val)}")
    print(f"Test samples:  {len(meta_test)}")

    print("\nTrain Columns:", list(meta_train.columns))

    print("\nTarget Distribution (Train):")
    print(
        meta_train[["azimuth", "zenith"]].describe().loc[["mean", "std", "min", "max"]]
    )

    # 3.2 Verify Split Ratio
    total_train_val = len(meta_train) + len(meta_val)
    train_ratio = len(meta_train) / total_train_val
    print(f"\nTrain split ratio: {train_ratio:.4f}")

    if not (0.79 <= train_ratio <= 0.81):
        raise AssertionError(
            f"Split ratio {train_ratio:.4f} deviates significantly from 0.80"
        )

    # 3.3 Verify No Leakage
    train_ids = set(meta_train["event_id"])
    val_ids = set(meta_val["event_id"])
    overlap = train_ids.intersection(val_ids)

    if len(overlap) > 0:
        raise AssertionError(
            f"Data Leakage detected! {len(overlap)} event_ids found in both train and val."
        )
    print("Split verification passed: No event_id overlap between train and val.")

    # 3.4 Verify File Paths
    def verify_paths(df, dataset_name):
        print(f"\nChecking file paths for {dataset_name}...")
        # Sample 1000 paths (or all if less than 1000)
        sample_size = min(1000, len(df))
        sample_paths = df["file_path"].sample(n=sample_size, random_state=42).tolist()

        missing_count = 0
        missing_examples = []

        for rel_path in sample_paths:
            full_path = os.path.join(INPUT_DIR, rel_path)
            if not os.path.exists(full_path):
                missing_count += 1
                if len(missing_examples) < 5:
                    missing_examples.append(rel_path)

        missing_ratio = missing_count / sample_size
        print(
            f"Missing file ratio: {missing_ratio:.4f} ({missing_count}/{sample_size})"
        )

        if missing_ratio > 0.5:
            print("Examples of missing paths:")
            for p in missing_examples:
                print(f"  - {p}")
            raise FileNotFoundError(
                f"Validation failed: >50% of file paths in {dataset_name} do not resolve."
            )

    verify_paths(meta_train, "Train Metadata")
    verify_paths(meta_val, "Val Metadata")
    verify_paths(meta_test, "Test Metadata")

    print("\nAll metadata generation and validation steps completed successfully.")


if __name__ == "__main__":
    main()
