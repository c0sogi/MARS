import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit


def main():
    # 1. Configuration
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    RANDOM_STATE = 42
    VAL_SIZE = 0.2

    # Ensure metadata directory exists
    os.makedirs(METADATA_DIR, exist_ok=True)

    # 2. Load Data
    print("Loading raw data from ./input...")
    train_path = os.path.join(INPUT_DIR, "train.csv")
    test_path = os.path.join(INPUT_DIR, "test.csv")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"File not found: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"File not found: {test_path}")

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    print(f"Original Train shape: {df_train.shape}")
    print(f"Original Test shape: {df_test.shape}")

    # 3. Split Data (Group Sampling)
    # We use GroupShuffleSplit based on 'context' to prevent leakage.
    # Questions from the same context/passage must stay in the same split.
    print("Splitting training data (Grouped by 'context')...")

    splitter = GroupShuffleSplit(
        n_splits=1, test_size=VAL_SIZE, random_state=RANDOM_STATE
    )

    # Get indices for the split
    train_idx, val_idx = next(splitter.split(df_train, groups=df_train["context"]))

    df_train_split = df_train.iloc[train_idx].copy()
    df_val_split = df_train.iloc[val_idx].copy()

    # 4. Save Metadata
    # Saving the actual split dataframes as metadata CSVs
    print("Saving metadata to ./metadata...")
    meta_train_path = os.path.join(METADATA_DIR, "train.csv")
    meta_val_path = os.path.join(METADATA_DIR, "val.csv")
    meta_test_path = os.path.join(METADATA_DIR, "test.csv")

    df_train_split.to_csv(meta_train_path, index=False)
    df_val_split.to_csv(meta_val_path, index=False)
    df_test.to_csv(meta_test_path, index=False)

    print("Metadata generation complete.")

    # 5. Verification
    print("\n=== Verification Steps ===")

    # Reload datasets
    d_train = pd.read_csv(meta_train_path)
    d_val = pd.read_csv(meta_val_path)
    d_test = pd.read_csv(meta_test_path)

    # A. Summary Statistics
    for name, df in [("Train", d_train), ("Validation", d_val), ("Test", d_test)]:
        print(f"\n[{name} Set]")
        print(f"Shape: {df.shape}")
        if "language" in df.columns:
            print("Language Distribution:")
            print(df["language"].value_counts(normalize=True))

    # B. Verify Group Split (Leakage Check)
    train_contexts = set(d_train["context"].unique())
    val_contexts = set(d_val["context"].unique())
    intersection = train_contexts.intersection(val_contexts)

    if len(intersection) > 0:
        raise AssertionError(
            f"Validation failed: Group leakage detected. {len(intersection)} contexts appear in both train and validation sets."
        )
    else:
        print("\n[Check Passed] No context leakage between train and validation sets.")

    # C. Verify Split Ratio
    n_train = len(d_train)
    n_val = len(d_val)
    total = n_train + n_val
    actual_val_ratio = n_val / total
    print(f"\nValidation Ratio: {actual_val_ratio:.4f} (Target: {VAL_SIZE})")

    # Allow a small margin of error due to group sizes
    if not (0.15 <= actual_val_ratio <= 0.25):
        print(
            "Warning: Validation ratio deviates significantly from 0.2 due to large group sizes."
        )

    # D. File Path Check
    # Check if any column looks like a file path and verify existence relative to ./input
    print("\nChecking for file path columns...")
    path_columns = []

    # Heuristic: Check first row for path-like strings
    if len(d_train) > 0:
        for col in d_train.columns:
            if d_train[col].dtype == object:
                sample = str(d_train[col].iloc[0])
                # Check for common path indicators
                if (
                    sample.startswith("./")
                    or sample.startswith("/")
                    or sample.lower().endswith((".jpg", ".png", ".wav", ".mp3", ".txt"))
                ):
                    # Exclude long text fields like context/question which might accidentally trigger
                    if len(sample) < 256:
                        path_columns.append(col)

    if path_columns:
        print(f"Found potential path columns: {path_columns}")
        for col in path_columns:
            # Check 1000 random samples
            sample_df = d_train[col].sample(
                n=min(1000, len(d_train)), random_state=RANDOM_STATE
            )
            missing_count = 0
            missing_examples = []

            for path_val in sample_df:
                # Paths in metadata are relative to ./input
                full_path = os.path.join(INPUT_DIR, str(path_val))
                if not os.path.exists(full_path):
                    missing_count += 1
                    if len(missing_examples) < 5:
                        missing_examples.append(full_path)

            ratio = missing_count / len(sample_df)
            print(f"Column '{col}': Missing File Ratio = {ratio:.4f}")

            if ratio > 0.5:
                print(f"Examples of missing files: {missing_examples}")
                raise FileNotFoundError(
                    f"Validation failed: More than 50% of files missing for column '{col}'."
                )
    else:
        print("No file path columns detected. Skipping file existence check.")

    print("\nAll checks passed successfully.")


if __name__ == "__main__":
    main()
