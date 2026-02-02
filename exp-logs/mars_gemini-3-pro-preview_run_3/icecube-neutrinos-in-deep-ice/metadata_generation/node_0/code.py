import pandas as pd
import numpy as np
from pathlib import Path
import os
from sklearn.model_selection import train_test_split

# Constants
INPUT_DIR = Path("./input")
METADATA_DIR = Path("./metadata")
RANDOM_STATE = 42
TRAIN_VAL_SPLIT_RATIO = 0.2


def main():
    print("Starting metadata generation...")

    # Ensure metadata directory exists
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # 1. Process Training Data
    # ---------------------------------------------------------
    train_meta_path = INPUT_DIR / "train_meta.parquet"
    print(f"Loading {train_meta_path}...")

    # Load training metadata
    train_df = pd.read_parquet(train_meta_path)

    # Construct relative file paths for batch files
    # Format: train/batch_{batch_id}.parquet
    print("Constructing file paths for training data...")
    train_df["file_path"] = (
        "train/batch_" + train_df["batch_id"].astype(str) + ".parquet"
    )

    # Perform 80/20 Train/Validation Split
    # Since this is a regression task and events are independent, we use a random split.
    print(
        f"Splitting data into Train (80%) and Validation (20%) with random_state={RANDOM_STATE}..."
    )
    train_split, val_split = train_test_split(
        train_df,
        test_size=TRAIN_VAL_SPLIT_RATIO,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    # Save to metadata directory
    print("Saving train_metadata.parquet...")
    train_split.to_parquet(METADATA_DIR / "train_metadata.parquet", index=False)

    print("Saving val_metadata.parquet...")
    val_split.to_parquet(METADATA_DIR / "val_metadata.parquet", index=False)

    # Free memory
    del train_df, train_split, val_split

    # ---------------------------------------------------------
    # 2. Process Test Data
    # ---------------------------------------------------------
    test_meta_path = INPUT_DIR / "test_meta.parquet"
    print(f"Loading {test_meta_path}...")

    # Load test metadata
    test_df = pd.read_parquet(test_meta_path)

    # Construct relative file paths for batch files
    # Format: test/batch_{batch_id}.parquet
    print("Constructing file paths for test data...")
    test_df["file_path"] = "test/batch_" + test_df["batch_id"].astype(str) + ".parquet"

    # Save to metadata directory
    print("Saving test_metadata.parquet...")
    test_df.to_parquet(METADATA_DIR / "test_metadata.parquet", index=False)

    # Free memory
    del test_df

    # ---------------------------------------------------------
    # 3. Verification & Checks
    # ---------------------------------------------------------
    print("\n--- Performing Verification Checks ---")

    datasets = {
        "Train": METADATA_DIR / "train_metadata.parquet",
        "Validation": METADATA_DIR / "val_metadata.parquet",
        "Test": METADATA_DIR / "test_metadata.parquet",
    }

    loaded_dfs = {}

    for name, path in datasets.items():
        print(f"\nChecking {name} dataset ({path})...")
        df = pd.read_parquet(path)
        loaded_dfs[name] = df

        # 3.1 Summary Statistics
        print(f"  Shape: {df.shape}")
        print(f"  Unique Events: {df['event_id'].nunique()}")
        if "azimuth" in df.columns:
            print(
                f"  Azimuth - Mean: {df['azimuth'].mean():.4f}, Std: {df['azimuth'].std():.4f}"
            )
            print(
                f"  Zenith  - Mean: {df['zenith'].mean():.4f}, Std: {df['zenith'].std():.4f}"
            )

        # 3.2 File Path Verification
        print("  Verifying file paths...")
        # Sample 1000 paths randomly
        sample_size = min(1000, len(df))
        sample_paths = (
            df["file_path"].sample(n=sample_size, random_state=RANDOM_STATE).tolist()
        )

        missing_count = 0
        missing_samples = []

        for rel_path in sample_paths:
            full_path = INPUT_DIR / rel_path
            if not full_path.exists():
                missing_count += 1
                if len(missing_samples) < 5:
                    missing_samples.append(str(rel_path))

        missing_ratio = missing_count / sample_size
        print(
            f"  Missing file ratio: {missing_ratio:.4f} ({missing_count}/{sample_size})"
        )

        if missing_ratio > 0.5:
            print(f"  [ERROR] Sample missing paths: {missing_samples}")
            raise FileNotFoundError(
                f"Missing file ratio for {name} is {missing_ratio}, which exceeds 0.5."
            )

    # 3.3 Split Verification
    print("\nVerifying Train/Validation Split...")
    train_ids = set(loaded_dfs["Train"]["event_id"])
    val_ids = set(loaded_dfs["Validation"]["event_id"])

    # Check overlap
    overlap = train_ids.intersection(val_ids)
    print(f"  Overlap between Train and Validation: {len(overlap)} events")
    if len(overlap) > 0:
        raise AssertionError(
            f"Data leakage detected! {len(overlap)} events are in both train and validation sets."
        )

    # Check ratio
    total_train_val = len(loaded_dfs["Train"]) + len(loaded_dfs["Validation"])
    actual_val_ratio = len(loaded_dfs["Validation"]) / total_train_val
    print(
        f"  Actual Validation Ratio: {actual_val_ratio:.4f} (Target: {TRAIN_VAL_SPLIT_RATIO})"
    )

    # Allow small tolerance for ratio
    if not (0.19 < actual_val_ratio < 0.21):
        raise AssertionError(
            f"Validation split ratio {actual_val_ratio:.4f} is significantly different from expected 0.2."
        )

    print("\nAll checks passed successfully. Metadata generation complete.")


if __name__ == "__main__":
    main()
