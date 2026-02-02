import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Configuration
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
RANDOM_STATE = 42
VAL_SIZE = 0.2
LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


def generate_metadata():
    """Generates metadata for train, val, and test sets."""
    os.makedirs(METADATA_DIR, exist_ok=True)

    # --- Process Training Data ---
    train_path = os.path.join(INPUT_DIR, "train.csv")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"{train_path} not found.")

    # Read CSV. We assume standard quoting for text fields with newlines.
    df_train = pd.read_csv(train_path)

    # Add source information (No copy of text, just reference)
    df_train["source_file"] = "train.csv"
    df_train["source_row_index"] = df_train.index

    # Ensure labels are integers and handle NaNs (though unlikely in this dataset)
    for col in LABEL_COLS:
        df_train[col] = df_train[col].fillna(0).astype(int)

    # Create Stratification Key
    # Concatenate all labels to form a unique string key for each combination
    stratify_key = df_train[LABEL_COLS].astype(str).agg("".join, axis=1)

    # Handle rare classes for stratification
    # If a combination appears < 2 times, we can't stratify on it.
    key_counts = stratify_key.value_counts()
    rare_keys = key_counts[key_counts < 2].index

    # Map rare keys to a common 'rare' bucket to attempt stratification
    stratify_key_adjusted = stratify_key.apply(
        lambda x: "rare" if x in rare_keys else x
    )

    # If 'rare' bucket itself has < 2 samples, we fallback to random split
    if (stratify_key_adjusted == "rare").sum() == 1:
        print(
            "Warning: 'rare' class bucket has only 1 sample. Falling back to random split."
        )
        stratify_final = None
    else:
        stratify_final = stratify_key_adjusted

    print("Splitting training data...")
    try:
        train_df, val_df = train_test_split(
            df_train,
            test_size=VAL_SIZE,
            random_state=RANDOM_STATE,
            stratify=stratify_final,
        )
    except ValueError as e:
        print(f"Stratification failed ({e}). Falling back to random split.")
        train_df, val_df = train_test_split(
            df_train, test_size=VAL_SIZE, random_state=RANDOM_STATE
        )

    # Define columns to save in metadata
    # We include ID, source reference, and labels. Text is NOT included to avoid duplication.
    meta_cols = ["id", "source_file", "source_row_index"] + LABEL_COLS

    # Save to metadata directory
    train_df[meta_cols].to_csv(
        os.path.join(METADATA_DIR, "train_metadata.csv"), index=False
    )
    val_df[meta_cols].to_csv(
        os.path.join(METADATA_DIR, "val_metadata.csv"), index=False
    )

    # --- Process Test Data ---
    test_path = os.path.join(INPUT_DIR, "test.csv")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"{test_path} not found.")

    df_test = pd.read_csv(test_path)
    df_test["source_file"] = "test.csv"
    df_test["source_row_index"] = df_test.index

    # Test metadata columns (Labels usually don't exist in test, but we check)
    test_meta_cols = ["id", "source_file", "source_row_index"]
    for col in LABEL_COLS:
        if col in df_test.columns:
            test_meta_cols.append(col)

    df_test[test_meta_cols].to_csv(
        os.path.join(METADATA_DIR, "test_metadata.csv"), index=False
    )

    print("Metadata generation complete.")


def check_file_paths(df, dataset_name):
    """Checks if source files referenced in metadata exist."""
    print(f"Checking file paths for {dataset_name}...")
    if "source_file" not in df.columns:
        return

    # Sample 1000 paths
    sample_size = min(1000, len(df))
    sample = df.sample(n=sample_size, random_state=42)

    missing_paths = []
    for _, row in sample.iterrows():
        # Path is relative to INPUT_DIR
        full_path = os.path.join(INPUT_DIR, row["source_file"])
        if not os.path.exists(full_path):
            missing_paths.append(row["source_file"])

    missing_ratio = len(missing_paths) / sample_size
    print(f"Missing file ratio: {missing_ratio:.4f}")

    if missing_ratio > 0.5:
        print("Sample missing paths:", missing_paths[:5])
        raise FileNotFoundError(
            f"More than 50% of file paths in {dataset_name} are missing."
        )


def validate_metadata():
    """Performs validation checks on generated metadata."""
    print("\n--- Validating Metadata ---")

    # Load metadata
    try:
        train_meta = pd.read_csv(os.path.join(METADATA_DIR, "train_metadata.csv"))
        val_meta = pd.read_csv(os.path.join(METADATA_DIR, "val_metadata.csv"))
        test_meta = pd.read_csv(os.path.join(METADATA_DIR, "test_metadata.csv"))
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Metadata files not found: {e}")

    # 1. Summary Statistics
    print(f"Train set shape: {train_meta.shape}")
    print(f"Validation set shape: {val_meta.shape}")
    print(f"Test set shape: {test_meta.shape}")

    print("\nTrain Label Distribution:")
    print(train_meta[LABEL_COLS].mean())
    print("\nValidation Label Distribution:")
    print(val_meta[LABEL_COLS].mean())

    # 2. Check File Paths
    check_file_paths(train_meta, "train")
    check_file_paths(val_meta, "validation")
    check_file_paths(test_meta, "test")

    # 3. Verify Stratification
    print("\nVerifying Stratification...")
    for col in LABEL_COLS:
        train_mean = train_meta[col].mean()
        val_mean = val_meta[col].mean()
        diff = abs(train_mean - val_mean)

        # Check if difference is acceptable.
        # For multi-label stratification, small deviations are expected but should be minimal.
        # We use a threshold of 0.015 (1.5%) absolute difference.
        if diff > 0.015:
            # We raise an error if the deviation is significant, indicating poor split.
            raise AssertionError(
                f"Stratification failed for column '{col}'. "
                f"Train mean: {train_mean:.4f}, Val mean: {val_mean:.4f}, Diff: {diff:.4f}"
            )

    print("Stratification verification passed.")
    print("All validation checks passed.")


if __name__ == "__main__":
    generate_metadata()
    validate_metadata()
