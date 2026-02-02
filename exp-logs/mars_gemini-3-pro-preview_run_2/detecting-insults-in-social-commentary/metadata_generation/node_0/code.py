import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split

# Configuration
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
RANDOM_STATE = 42
VAL_SIZE = 0.2


def load_data(filename):
    """Loads a CSV file and cleans up index columns if present."""
    path = os.path.join(INPUT_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")

    df = pd.read_csv(path)

    # Remove unnamed index column if it exists (common in pandas saved csvs)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    return df


def generate_metadata():
    print("Generating metadata...")
    os.makedirs(METADATA_DIR, exist_ok=True)

    # 1. Load Data
    try:
        df_train_full = load_data("train.csv")
        df_test = load_data("test.csv")
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    # 2. Split Training Data
    # We need to stratify based on the 'Insult' column
    if "Insult" not in df_train_full.columns:
        raise ValueError("train.csv does not contain 'Insult' column.")

    X = df_train_full
    y = df_train_full["Insult"]

    # Perform stratified split
    df_train, df_val = train_test_split(
        X, test_size=VAL_SIZE, stratify=y, random_state=RANDOM_STATE, shuffle=True
    )

    # 3. Save Metadata
    # We save the dataframes directly to metadata.
    # This serves as the definition for train/val/test sets for downstream tasks.
    train_save_path = os.path.join(METADATA_DIR, "train.csv")
    val_save_path = os.path.join(METADATA_DIR, "val.csv")
    test_save_path = os.path.join(METADATA_DIR, "test.csv")

    df_train.to_csv(train_save_path, index=False)
    df_val.to_csv(val_save_path, index=False)
    df_test.to_csv(test_save_path, index=False)

    print(f"Saved train metadata to {train_save_path}")
    print(f"Saved val metadata to {val_save_path}")
    print(f"Saved test metadata to {test_save_path}")


def check_file_paths(df, name):
    """
    Checks if columns look like file paths and verifies them.
    For this text dataset, this is likely not applicable, but implemented for robustness.
    """
    # Heuristic to detect path columns: strings containing '/' or ending in common extensions
    path_cols = []
    if len(df) > 0:
        for col in df.columns:
            if df[col].dtype == object:
                sample = str(df[col].iloc[0])
                if "/" in sample or sample.lower().endswith(
                    (".jpg", ".png", ".wav", ".csv")
                ):
                    path_cols.append(col)

    if not path_cols:
        return

    print(f"Checking file paths in {name} for columns: {path_cols}")
    for col in path_cols:
        # Sample 1000 paths
        sample_paths = (
            df[col].sample(n=min(1000, len(df)), random_state=RANDOM_STATE).tolist()
        )
        missing_count = 0
        missing_samples = []

        for p in sample_paths:
            # Paths in metadata must be relative to ./input
            full_path = os.path.join(INPUT_DIR, str(p))
            if not os.path.exists(full_path):
                missing_count += 1
                if len(missing_samples) < 5:
                    missing_samples.append(p)

        missing_ratio = missing_count / len(sample_paths)
        if missing_ratio > 0.5:
            print(f"Sample missing paths: {missing_samples}")
            raise FileNotFoundError(
                f"More than 50% of files missing in column {col} of {name}. Ratio: {missing_ratio}"
            )


def validate_metadata():
    print("\nValidating generated metadata...")

    # Load the generated files
    try:
        df_train = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
        df_val = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))
        df_test = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))
    except FileNotFoundError as e:
        raise AssertionError(f"Metadata file missing: {e}")

    # 1. Summary Statistics
    print("-" * 30)
    print(f"Train Rows: {len(df_train)}")
    print(f"Val Rows:   {len(df_val)}")
    print(f"Test Rows:  {len(df_test)}")
    print("-" * 30)

    if "Insult" in df_train.columns:
        print("Train Class Distribution:")
        print(df_train["Insult"].value_counts(normalize=True))

    if "Insult" in df_val.columns:
        print("Val Class Distribution:")
        print(df_val["Insult"].value_counts(normalize=True))
    print("-" * 30)

    # 2. Check File Paths (if applicable)
    # This dataset is text-based (Date, Comment), so this will likely skip.
    check_file_paths(df_train, "train.csv")
    check_file_paths(df_val, "val.csv")
    check_file_paths(df_test, "test.csv")

    # 3. Verify Split Requirements
    # Check Split Ratio
    n_train = len(df_train)
    n_val = len(df_val)
    total = n_train + n_val
    val_ratio = n_val / total

    print(f"Actual Validation Ratio: {val_ratio:.4f}")

    # Assert ratio is close to target (0.2)
    # Allow small deviation due to integer rounding
    if not (0.19 <= val_ratio <= 0.21):
        raise AssertionError(
            f"Validation split ratio {val_ratio:.4f} is not within acceptable range (0.19-0.21)"
        )

    # Check Stratification
    # We expect the mean of the target to be roughly the same
    if "Insult" in df_train.columns and "Insult" in df_val.columns:
        train_mean = df_train["Insult"].mean()
        val_mean = df_val["Insult"].mean()
        diff = abs(train_mean - val_mean)

        print(f"Train Target Mean: {train_mean:.4f}")
        print(f"Val Target Mean:   {val_mean:.4f}")
        print(f"Difference:        {diff:.4f}")

        # Assert difference is small (e.g., < 2%)
        if diff > 0.02:
            raise AssertionError(
                f"Stratification failed. Target distribution difference {diff:.4f} is too high."
            )

    print("\nAll validation checks passed successfully.")


if __name__ == "__main__":
    generate_metadata()
    validate_metadata()
