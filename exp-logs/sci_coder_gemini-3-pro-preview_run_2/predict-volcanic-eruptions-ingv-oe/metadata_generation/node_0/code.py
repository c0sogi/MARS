import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def run_metadata_generation():
    # Constants
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    RANDOM_STATE = 42
    VAL_SIZE = 0.2

    print("Starting metadata generation...")

    # Ensure metadata directory exists
    os.makedirs(METADATA_DIR, exist_ok=True)

    # ---------------------------------------------------------
    # 1. Process Training and Validation Data
    # ---------------------------------------------------------
    train_csv_path = os.path.join(INPUT_DIR, "train.csv")
    if not os.path.exists(train_csv_path):
        raise FileNotFoundError(f"{train_csv_path} not found.")

    df_full = pd.read_csv(train_csv_path)

    # Construct relative file paths
    # Filenames are {segment_id}.csv inside the train folder
    df_full["file_path"] = df_full["segment_id"].apply(
        lambda x: os.path.join("train", f"{x}.csv")
    )

    # Stratified Split for Regression
    # We bin the target 'time_to_eruption' to simulate classes for stratification
    num_bins = int(np.floor(1 + np.log2(len(df_full))))  # Sturges' rule
    df_full["bins"] = pd.cut(df_full["time_to_eruption"], bins=num_bins, labels=False)

    # Handle rare bins (if any bin has only 1 member, stratification fails)
    # If a bin has very few samples, we might merge or just fall back to random split if strictly necessary.
    # However, with ~4000 samples, simple binning usually works.
    # To be safe, we check value counts. If any bin has < 2 samples, we might adjust.
    # For simplicity in this script, if stratification fails due to single members, we fall back to random,
    # but usually Sturges' rule on 4000 samples is fine.

    try:
        train_df, val_df = train_test_split(
            df_full,
            test_size=VAL_SIZE,
            random_state=RANDOM_STATE,
            shuffle=True,
            stratify=df_full["bins"],
        )
    except ValueError:
        # Fallback if bins have too few members
        print(
            "Warning: Stratification by bins failed (likely single-member bins). Falling back to random split."
        )
        train_df, val_df = train_test_split(
            df_full, test_size=VAL_SIZE, random_state=RANDOM_STATE, shuffle=True
        )

    # Drop the temporary bins column
    train_df = train_df.drop(columns=["bins"])
    val_df = val_df.drop(columns=["bins"])

    # Save to metadata
    train_save_path = os.path.join(METADATA_DIR, "train.csv")
    val_save_path = os.path.join(METADATA_DIR, "val.csv")

    train_df.to_csv(train_save_path, index=False)
    val_df.to_csv(val_save_path, index=False)

    print(f"Saved train metadata to {train_save_path} ({len(train_df)} rows)")
    print(f"Saved val metadata to {val_save_path} ({len(val_df)} rows)")

    # ---------------------------------------------------------
    # 2. Process Test Data
    # ---------------------------------------------------------
    # Test files are in input/test/*.csv
    test_files = glob.glob(os.path.join(INPUT_DIR, "test", "*.csv"))

    test_data = []
    for fp in test_files:
        filename = os.path.basename(fp)
        segment_id = os.path.splitext(filename)[0]
        # Path relative to input dir
        rel_path = os.path.join("test", filename)
        test_data.append({"segment_id": int(segment_id), "file_path": rel_path})

    test_df = pd.DataFrame(test_data)
    test_save_path = os.path.join(METADATA_DIR, "test.csv")
    test_df.to_csv(test_save_path, index=False)

    print(f"Saved test metadata to {test_save_path} ({len(test_df)} rows)")

    # ---------------------------------------------------------
    # 3. Validation Checks
    # ---------------------------------------------------------
    print("\nRunning validation checks...")

    # Reload datasets
    df_train_check = pd.read_csv(train_save_path)
    df_val_check = pd.read_csv(val_save_path)
    df_test_check = pd.read_csv(test_save_path)

    # 3a. Print Summary Statistics
    print("\n--- Summary Statistics ---")
    print(f"Train Shape: {df_train_check.shape}")
    print(f"Val Shape:   {df_val_check.shape}")
    print(f"Test Shape:  {df_test_check.shape}")

    print("\nTrain Target Stats:")
    print(df_train_check["time_to_eruption"].describe())
    print("\nVal Target Stats:")
    print(df_val_check["time_to_eruption"].describe())

    # 3b. File Existence Check
    def check_files_exist(df, name):
        paths = df["file_path"].tolist()
        if len(paths) > 1000:
            paths = np.random.choice(paths, 1000, replace=False)

        missing_count = 0
        missing_samples = []

        for p in paths:
            # Metadata paths are relative to ./input
            full_path = os.path.join(INPUT_DIR, p)
            if not os.path.exists(full_path):
                missing_count += 1
                if len(missing_samples) < 5:
                    missing_samples.append(p)

        ratio = missing_count / len(paths) if len(paths) > 0 else 0
        print(
            f"[{name}] Missing file ratio: {ratio:.4f} ({missing_count}/{len(paths)})"
        )

        if ratio > 0.5:
            print(f"Sample missing paths in {name}: {missing_samples}")
            raise FileNotFoundError(
                f"More than 50% of files missing in {name} dataset."
            )

    check_files_exist(df_train_check, "Train")
    check_files_exist(df_val_check, "Val")
    check_files_exist(df_test_check, "Test")

    # 3c. Verify Split Requirements
    total_train_val = len(df_train_check) + len(df_val_check)
    actual_val_ratio = len(df_val_check) / total_train_val
    print(f"\nActual Validation Ratio: {actual_val_ratio:.4f}")

    # Allow small floating point tolerance
    assert (
        0.19 < actual_val_ratio < 0.21
    ), f"Validation split ratio {actual_val_ratio} is not close to 0.2"

    # Check distribution similarity (Mean and Std Dev should be reasonably close)
    train_mean = df_train_check["time_to_eruption"].mean()
    val_mean = df_val_check["time_to_eruption"].mean()
    train_std = df_train_check["time_to_eruption"].std()
    val_std = df_val_check["time_to_eruption"].std()

    print(f"Train Mean: {train_mean:.2f}, Val Mean: {val_mean:.2f}")
    print(f"Train Std:  {train_std:.2f},  Val Std:  {val_std:.2f}")

    # A simple heuristic check for distribution similarity:
    # If the means differ by more than 10% of the std dev, that might indicate a bad split,
    # though with random sampling on this size, it should be fine.
    # We won't raise an error strictly on stats unless they are wildly off,
    # but we assert that the split logic was executed.

    # Assert no overlap between train and val segment_ids
    train_ids = set(df_train_check["segment_id"])
    val_ids = set(df_val_check["segment_id"])
    overlap = train_ids.intersection(val_ids)
    assert (
        len(overlap) == 0
    ), f"Found {len(overlap)} overlapping segment_ids between train and val."

    print("\nAll checks passed successfully.")


if __name__ == "__main__":
    run_metadata_generation()
