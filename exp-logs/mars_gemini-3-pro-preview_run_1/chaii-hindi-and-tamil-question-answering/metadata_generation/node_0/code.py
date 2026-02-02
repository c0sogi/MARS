import pandas as pd
import numpy as np
import os
from sklearn.model_selection import GroupShuffleSplit

# Configuration
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
RANDOM_STATE = 42
TRAIN_SIZE = 0.8


def check_file_paths(df, name):
    """
    Checks if columns contain file paths relative to input and verifies their existence.
    """
    # Heuristic to identify path columns: string type and contains '/'
    path_cols = []
    for col in df.columns:
        if df[col].dtype == "object":
            # Check a sample
            sample = df[col].dropna().iloc[0] if not df[col].dropna().empty else ""
            if isinstance(sample, str) and ("/" in sample or sample.startswith(".")):
                # Further check if it looks like a file path (has extension)
                if "." in os.path.basename(sample):
                    path_cols.append(col)

    if not path_cols:
        print(f"[{name}] No file path columns detected. Skipping file existence check.")
        return

    print(f"[{name}] Checking file paths in columns: {path_cols}")

    for col in path_cols:
        # Select 1000 random samples or all if less than 1000
        samples = (
            df[col].dropna().sample(n=min(1000, len(df)), random_state=RANDOM_STATE)
        )
        missing_count = 0
        missing_samples = []

        for path in samples:
            # Paths should be relative to input or absolute.
            # Assuming paths in metadata might be relative to ./input if generated that way,
            # or relative to working dir. The prompt says "relative to the ./input directory".
            # We construct the full path.
            full_path = os.path.join(INPUT_DIR, path)
            if not os.path.exists(full_path):
                # Try checking if it is relative to CWD
                if not os.path.exists(path):
                    missing_count += 1
                    if len(missing_samples) < 5:
                        missing_samples.append(path)

        missing_ratio = missing_count / len(samples)
        print(f"  Column '{col}': Missing Ratio = {missing_ratio:.4f}")

        if missing_ratio > 0.5:
            print(f"  Sample missing paths: {missing_samples}")
            raise FileNotFoundError(
                f"More than 50% of files missing in column {col} for {name}"
            )


def main():
    # 1. Setup
    if not os.path.exists(METADATA_DIR):
        os.makedirs(METADATA_DIR)

    print("Loading raw data...")
    try:
        train_df = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))
        test_df = pd.read_csv(os.path.join(INPUT_DIR, "test.csv"))
    except FileNotFoundError as e:
        print(f"Error loading input files: {e}")
        return

    # 2. Group Split
    # We must split by context to prevent leakage.
    print("Splitting data...")
    splitter = GroupShuffleSplit(
        n_splits=1, train_size=TRAIN_SIZE, random_state=RANDOM_STATE
    )

    # The groups are defined by the unique context
    train_idx, val_idx = next(splitter.split(train_df, groups=train_df["context"]))

    new_train_df = train_df.iloc[train_idx].copy()
    new_val_df = train_df.iloc[val_idx].copy()

    # 3. Save Metadata
    print("Saving metadata...")
    new_train_df.to_csv(os.path.join(METADATA_DIR, "train.csv"), index=False)
    new_val_df.to_csv(os.path.join(METADATA_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(METADATA_DIR, "test.csv"), index=False)

    print("Metadata generation complete. Starting validation...")

    # 4. Validation

    # Reload datasets
    val_train = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
    val_val = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))
    val_test = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))

    # Summary Statistics
    print("\n==== Summary Statistics ====")
    print(f"Train Rows: {len(val_train)}")
    print(f"Val Rows:   {len(val_val)}")
    print(f"Test Rows:  {len(val_test)}")

    print("\nTrain Language Distribution:")
    print(val_train["language"].value_counts())
    print("\nVal Language Distribution:")
    print(val_val["language"].value_counts())

    # File Path Checks
    check_file_paths(val_train, "Train")
    check_file_paths(val_val, "Val")
    check_file_paths(val_test, "Test")

    # Verify Split Logic
    print("\n==== Verifying Split Logic ====")

    # Check 1: Ratio
    total_samples = len(val_train) + len(val_val)
    actual_ratio = len(val_train) / total_samples
    print(f"Target Ratio: {TRAIN_SIZE}, Actual Ratio: {actual_ratio:.4f}")

    # We allow a small margin of error because group splitting cannot be exact
    if abs(actual_ratio - TRAIN_SIZE) > 0.05:
        print(
            "Warning: Split ratio deviates by more than 5% from target. This is expected if groups are large/imbalanced."
        )

    # Check 2: Leakage (Group Integrity)
    train_contexts = set(val_train["context"].unique())
    val_contexts = set(val_val["context"].unique())

    intersection = train_contexts.intersection(val_contexts)
    leakage_count = len(intersection)

    print(f"Context intersection count: {leakage_count}")

    if leakage_count > 0:
        raise AssertionError(
            f"Data Leakage Detected! {leakage_count} contexts appear in both train and validation sets."
        )

    print("Validation successful: No context leakage detected.")


if __name__ == "__main__":
    main()
