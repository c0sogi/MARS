import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Constants
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
TRAIN_CSV = os.path.join(INPUT_DIR, "train.csv")
TEST_CSV = os.path.join(INPUT_DIR, "test.csv")
RANDOM_STATE = 42
VAL_SIZE = 0.2


def generate_metadata():
    # Create metadata directory
    os.makedirs(METADATA_DIR, exist_ok=True)

    # Load raw data
    print("Loading raw data...")
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    # Generate file paths relative to ./input
    # Assuming the directory structure follows the pattern: input/{train|test}/{id}/geometry.xyz
    print("Generating file paths...")
    train_df["file_path"] = train_df["id"].apply(lambda x: f"train/{x}/geometry.xyz")
    test_df["file_path"] = test_df["id"].apply(lambda x: f"test/{x}/geometry.xyz")

    # Stratified Split for Validation
    # Since this is a regression task, we bin the target variable to allow for stratified splitting
    print("Splitting training data into train/val...")

    # We will stratify based on 'formation_energy_ev_natom'
    # Create bins
    num_bins = 10
    train_df["stratify_bin"] = pd.qcut(
        train_df["formation_energy_ev_natom"],
        q=num_bins,
        labels=False,
        duplicates="drop",
    )

    # Split
    train_split, val_split = train_test_split(
        train_df,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=train_df["stratify_bin"],
    )

    # Drop the temporary stratification column
    train_split = train_split.drop(columns=["stratify_bin"])
    val_split = val_split.drop(columns=["stratify_bin"])
    # We don't need to drop it from test_df as it wasn't added

    # Save to metadata
    print("Saving metadata files...")
    train_split.to_csv(os.path.join(METADATA_DIR, "train_metadata.csv"), index=False)
    val_split.to_csv(os.path.join(METADATA_DIR, "val_metadata.csv"), index=False)
    test_df.to_csv(os.path.join(METADATA_DIR, "test_metadata.csv"), index=False)

    print("Metadata generation complete.")


def check_file_paths(df, name):
    print(f"Checking file paths for {name} dataset...")
    # Sample up to 1000 paths
    sample_size = min(len(df), 1000)
    sample_paths = df["file_path"].sample(n=sample_size, random_state=RANDOM_STATE)

    missing_count = 0
    missing_samples = []

    for rel_path in sample_paths:
        full_path = os.path.join(INPUT_DIR, rel_path)
        if not os.path.exists(full_path):
            missing_count += 1
            if len(missing_samples) < 5:
                missing_samples.append(rel_path)

    missing_ratio = missing_count / sample_size
    print(f"  Missing file ratio: {missing_ratio:.4f} ({missing_count}/{sample_size})")

    if missing_ratio > 0.5:
        print("  Sample missing paths:")
        for p in missing_samples:
            print(f"    {p}")
        raise FileNotFoundError(
            f"More than 50% of file paths in {name} metadata do not exist in {INPUT_DIR}."
        )


def validate_metadata():
    print("\nStarting validation...")

    # Load generated metadata
    train_meta = pd.read_csv(os.path.join(METADATA_DIR, "train_metadata.csv"))
    val_meta = pd.read_csv(os.path.join(METADATA_DIR, "val_metadata.csv"))
    test_meta = pd.read_csv(os.path.join(METADATA_DIR, "test_metadata.csv"))

    # 1. Summary Statistics
    print("\nSummary Statistics:")
    print("-" * 20)
    print(f"Train set size: {len(train_meta)}")
    print(f"Val set size:   {len(val_meta)}")
    print(f"Test set size:  {len(test_meta)}")

    print("\nTrain Targets Mean:")
    print(train_meta[["formation_energy_ev_natom", "bandgap_energy_ev"]].mean())
    print("\nVal Targets Mean:")
    print(val_meta[["formation_energy_ev_natom", "bandgap_energy_ev"]].mean())

    # 2. Check File Paths
    print("-" * 20)
    check_file_paths(train_meta, "Train")
    check_file_paths(val_meta, "Validation")
    check_file_paths(test_meta, "Test")

    # 3. Verify Split Requirements
    print("-" * 20)
    print("Verifying split requirements...")

    total_train_val = len(train_meta) + len(val_meta)
    val_ratio = len(val_meta) / total_train_val

    print(f"  Actual Validation Ratio: {val_ratio:.4f}")

    # Allow for small deviation due to rounding/binning
    if not (0.19 <= val_ratio <= 0.21):
        raise AssertionError(
            f"Validation split ratio {val_ratio:.4f} deviates significantly from 0.20"
        )

    # Verify stratification (simple check of means)
    train_mean = train_meta["formation_energy_ev_natom"].mean()
    val_mean = val_meta["formation_energy_ev_natom"].mean()
    train_std = train_meta["formation_energy_ev_natom"].std()
    val_std = val_meta["formation_energy_ev_natom"].std()

    print(f"  Train Energy Mean: {train_mean:.4f}, Std: {train_std:.4f}")
    print(f"  Val Energy Mean:   {val_mean:.4f}, Std: {val_std:.4f}")

    # Check if means are reasonably close (within 0.5 standard deviations is a loose but sanity-check heuristic)
    if abs(train_mean - val_mean) > 0.5 * train_std:
        raise AssertionError(
            "Stratification check failed: Means of training and validation sets are too different."
        )

    print("\nAll validation checks passed successfully.")


if __name__ == "__main__":
    generate_metadata()
    validate_metadata()
