import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import shutil

# Constants
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
RANDOM_STATE = 42
VAL_SIZE = 0.2


def create_metadata():
    """
    Reads raw data, performs stratified split, and saves metadata files.
    """
    print("Starting metadata generation...")

    # Ensure metadata directory exists
    if os.path.exists(METADATA_DIR):
        shutil.rmtree(METADATA_DIR)
    os.makedirs(METADATA_DIR)

    # Load raw data
    train_path = os.path.join(INPUT_DIR, "train.csv")
    test_path = os.path.join(INPUT_DIR, "test.csv")

    print(f"Reading {train_path}...")
    df_train_full = pd.read_csv(train_path)

    print(f"Reading {test_path}...")
    df_test = pd.read_csv(test_path)

    # Prepare stratification label
    # The targets are one-hot encoded. We convert them to a single label for stratification.
    # 0: model_a, 1: model_b, 2: tie
    def get_label(row):
        if row["winner_model_a"] == 1:
            return 0
        elif row["winner_model_b"] == 1:
            return 1
        elif row["winner_tie"] == 1:
            return 2
        else:
            # Fallback for unexpected rows (though dataset description implies valid one-hot)
            return -1

    df_train_full["stratify_label"] = df_train_full.apply(get_label, axis=1)

    # Check for any unlabeled rows
    if (df_train_full["stratify_label"] == -1).any():
        print(
            "Warning: Some rows in training data do not have a clear winner. Dropping them for split."
        )
        df_train_full = df_train_full[df_train_full["stratify_label"] != -1].copy()

    # Perform Stratified Split
    print("Splitting data into train and validation sets...")
    df_train, df_val = train_test_split(
        df_train_full,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=df_train_full["stratify_label"],
        shuffle=True,
    )

    # Drop the temporary stratification label column
    df_train = df_train.drop(columns=["stratify_label"])
    df_val = df_val.drop(columns=["stratify_label"])
    df_train_full = df_train_full.drop(columns=["stratify_label"])

    # Save metadata files
    train_meta_path = os.path.join(METADATA_DIR, "train.csv")
    val_meta_path = os.path.join(METADATA_DIR, "val.csv")
    test_meta_path = os.path.join(METADATA_DIR, "test.csv")

    print(f"Saving train metadata to {train_meta_path}...")
    df_train.to_csv(train_meta_path, index=False)

    print(f"Saving validation metadata to {val_meta_path}...")
    df_val.to_csv(val_meta_path, index=False)

    print(f"Saving test metadata to {test_meta_path}...")
    df_test.to_csv(test_meta_path, index=False)

    print("Metadata generation complete.")
    return train_meta_path, val_meta_path, test_meta_path


def validate_metadata(train_path, val_path, test_path):
    """
    Loads generated metadata and performs validation checks.
    """
    print("\nStarting validation of generated metadata...")

    # Load datasets
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    # 1. Summary Statistics
    print("\n--- Summary Statistics ---")
    print(f"Train set shape: {df_train.shape}")
    print(f"Val set shape:   {df_val.shape}")
    print(f"Test set shape:  {df_test.shape}")

    # Calculate class distributions
    def get_distribution(df, name):
        if "winner_model_a" not in df.columns:
            return None

        count_a = df["winner_model_a"].sum()
        count_b = df["winner_model_b"].sum()
        count_tie = df["winner_tie"].sum()
        total = len(df)

        dist = {
            "model_a": count_a / total,
            "model_b": count_b / total,
            "tie": count_tie / total,
        }
        print(f"\nClass distribution for {name}:")
        print(f"  Model A: {count_a} ({dist['model_a']:.4f})")
        print(f"  Model B: {count_b} ({dist['model_b']:.4f})")
        print(f"  Tie:     {count_tie} ({dist['tie']:.4f})")
        return dist

    train_dist = get_distribution(df_train, "Train")
    val_dist = get_distribution(df_val, "Validation")

    # 2. Check File Paths (If applicable)
    # The dataset description indicates text data in CSV columns, not external file paths.
    # However, we scan columns for string values that look like relative paths starting with 'input/' just in case.
    # Given the task description (chatbot prompts), it's unlikely, but we implement the logic as requested.

    print("\n--- Checking File Paths ---")

    # Helper to check paths
    def check_paths(df, name):
        # Identify columns that might contain paths (simple heuristic: contains '/')
        path_cols = []
        for col in df.columns:
            if df[col].dtype == object:
                # Check first non-null value
                sample = df[col].dropna().iloc[0] if not df[col].dropna().empty else ""
                if isinstance(sample, str) and ("/" in sample or "\\" in sample):
                    # It might be a path, but for this specific dataset (text prompts),
                    # prompts might contain slashes. We only check if it looks like a file path relative to input.
                    # As per instructions: "All file paths stored within the metadata must be relative to the ./input directory."
                    # We'll check if it starts with "./input" or "input/".
                    if sample.startswith("./input") or sample.startswith("input/"):
                        path_cols.append(col)

        if not path_cols:
            print(
                f"No file path columns detected in {name} metadata. Skipping path check."
            )
            return

        for col in path_cols:
            print(f"Checking column '{col}' in {name}...")
            paths = (
                df[col]
                .dropna()
                .sample(n=min(1000, len(df)), random_state=RANDOM_STATE)
                .tolist()
            )
            missing_count = 0
            missing_samples = []

            for p in paths:
                # Resolve relative to current working directory (since paths are relative to ./input or just relative)
                # If path is "input/file.txt", and we are in root, os.path.exists("input/file.txt") works.
                if not os.path.exists(p):
                    missing_count += 1
                    if len(missing_samples) < 5:
                        missing_samples.append(p)

            missing_ratio = missing_count / len(paths)
            print(f"  Missing file ratio for '{col}': {missing_ratio:.4f}")

            if missing_ratio > 0.5:
                print("  Sample missing paths:", missing_samples)
                raise FileNotFoundError(
                    f"More than 50% of files missing in column {col} of {name} dataset."
                )

    check_paths(df_train, "Train")
    check_paths(df_val, "Validation")
    check_paths(df_test, "Test")

    # 3. Verify Validation Split Requirements
    print("\n--- Verifying Split Requirements ---")

    # Check Ratio
    total_train_val = len(df_train) + len(df_val)
    val_ratio = len(df_val) / total_train_val
    print(f"Actual Validation Ratio: {val_ratio:.4f} (Target: {VAL_SIZE})")

    if not (0.19 < val_ratio < 0.21):
        raise AssertionError(
            f"Validation split ratio {val_ratio:.4f} is not close enough to target {VAL_SIZE}"
        )

    # Check Stratification
    # We compare the distribution dictionaries.
    # Allow a small tolerance since exact stratification isn't always possible with small counts (though here N is large).
    tolerance = 0.01
    for key in train_dist:
        diff = abs(train_dist[key] - val_dist[key])
        if diff > tolerance:
            raise AssertionError(
                f"Stratification failed for class {key}. Train: {train_dist[key]:.4f}, Val: {val_dist[key]:.4f}, Diff: {diff:.4f}"
            )

    print("Stratification check passed.")
    print("\nAll validation checks passed successfully.")


if __name__ == "__main__":
    try:
        train_meta, val_meta, test_meta = create_metadata()
        validate_metadata(train_meta, val_meta, test_meta)
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        exit(1)
