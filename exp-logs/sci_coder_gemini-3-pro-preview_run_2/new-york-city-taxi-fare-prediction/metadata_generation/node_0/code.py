import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def main():
    # Configuration
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    RANDOM_STATE = 42
    VAL_SIZE = 0.2

    # Ensure metadata directory exists
    os.makedirs(METADATA_DIR, exist_ok=True)

    print("Loading raw data...")
    # Load training data (labels.csv)
    # Using pyarrow engine for faster reading of large CSVs if available, otherwise default
    try:
        df = pd.read_csv(os.path.join(INPUT_DIR, "labels.csv"), engine="pyarrow")
    except (ValueError, ImportError):
        print("Pyarrow engine not available or failed, falling back to default.")
        df = pd.read_csv(os.path.join(INPUT_DIR, "labels.csv"))

    # Load test data
    try:
        test_df = pd.read_csv(os.path.join(INPUT_DIR, "test.csv"), engine="pyarrow")
    except (ValueError, ImportError):
        test_df = pd.read_csv(os.path.join(INPUT_DIR, "test.csv"))

    print(f"Original dataset shape: {df.shape}")
    print(f"Test dataset shape: {test_df.shape}")

    # Perform Split
    # Requirements: 80:20 split, random shuffle, fixed random state.
    # Not a classification task (Regression), so StratifiedKFold is not strictly applicable/required by prompt logic.
    # No inherent groups mentioned (key is unique per row), so GroupShuffleSplit is not applicable.
    print("Splitting data into training and validation sets...")
    train_df, val_df = train_test_split(
        df, test_size=VAL_SIZE, random_state=RANDOM_STATE, shuffle=True
    )

    # Save metadata (Parquet is efficient for tabular metadata/data)
    print("Saving metadata to ./metadata...")
    train_path = os.path.join(METADATA_DIR, "train.parquet")
    val_path = os.path.join(METADATA_DIR, "val.parquet")
    test_path = os.path.join(METADATA_DIR, "test.parquet")

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    test_df.to_parquet(test_path, index=False)

    # ---------------------------------------------------------
    # Verification Step
    # ---------------------------------------------------------
    print("Verifying generated metadata...")

    # Reload data
    t_meta = pd.read_parquet(train_path)
    v_meta = pd.read_parquet(val_path)
    test_meta = pd.read_parquet(test_path)

    # 1. Print Summary Statistics
    print("\n=== Summary Statistics ===")
    print(f"Train Set: {t_meta.shape[0]} samples")
    print(f"Val Set:   {v_meta.shape[0]} samples")
    print(f"Test Set:  {test_meta.shape[0]} samples")

    print("\nTrain Fare Amount Stats:")
    print(t_meta["fare_amount"].describe())
    print("\nVal Fare Amount Stats:")
    print(v_meta["fare_amount"].describe())

    # 2. Verify Split Requirements
    total_samples = len(df)
    expected_val_count = int(total_samples * VAL_SIZE)
    # Allow small rounding difference of 1
    assert (
        abs(len(v_meta) - expected_val_count) <= 1
    ), f"Validation set size mismatch. Expected approx {expected_val_count}, got {len(v_meta)}"

    assert (
        len(t_meta) + len(v_meta) == total_samples
    ), "Sum of train and val samples does not match original dataset size."

    # Check for data leakage (intersection of keys)
    # Using set intersection on a sample if dataset is too large, or full check if feasible.
    # 55M is large for set intersection in pure python, but pandas is optimized.
    # We trust train_test_split, but let's do a quick check on indices if preserved,
    # or just assume correctness from library.
    # To be rigorous but memory safe, we skip full set intersection of 55M strings here
    # unless necessary. train_test_split guarantees disjoint sets.

    # 3. File Path Verification
    # The dataset contains coordinates and timestamps, no external file paths.
    # We scan columns to see if any look like file paths (e.g. contain '/' and '.').
    # If found, we check them.

    print("\nChecking for file paths in metadata...")
    path_columns = []
    # Heuristic to detect path columns: string type and contains '/'
    for col in t_meta.select_dtypes(include=["object", "string"]).columns:
        # Check first valid value
        sample_val = (
            t_meta[col].dropna().iloc[0] if not t_meta[col].dropna().empty else ""
        )
        if isinstance(sample_val, str) and "/" in sample_val and "." in sample_val:
            # Likely a path
            path_columns.append(col)

    if path_columns:
        print(f"Detected potential file path columns: {path_columns}")
        # Check 1000 random paths
        missing_count = 0
        total_checks = 1000

        # Sample from train
        sample_paths = (
            t_meta[path_columns[0]].sample(n=total_checks, random_state=42).values
        )
        non_resolving_samples = []

        for p in sample_paths:
            # Paths must be relative to ./input
            full_path = os.path.join(INPUT_DIR, str(p))
            if not os.path.exists(full_path):
                missing_count += 1
                if len(non_resolving_samples) < 5:
                    non_resolving_samples.append(p)

        ratio = missing_count / total_checks
        print(f"Missing file ratio: {ratio:.4f}")

        if ratio > 0.5:
            print("Sample non-resolving paths:")
            for p in non_resolving_samples:
                print(f"  {p}")
            raise FileNotFoundError(
                f"More than 50% of file paths do not resolve. Ratio: {ratio}"
            )
    else:
        print("No file path columns detected. Skipping file existence check.")

    print("\nMetadata generation and verification complete.")


if __name__ == "__main__":
    main()
