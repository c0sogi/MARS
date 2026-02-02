import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Constants
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
RANDOM_STATE = 42
VAL_SIZE = 0.2


def generate_metadata():
    # Create metadata directory
    os.makedirs(METADATA_DIR, exist_ok=True)

    # File paths
    train_json_path = os.path.join(INPUT_DIR, "train.json")
    test_json_path = os.path.join(INPUT_DIR, "test.json")

    # --- Process Training Data ---
    print("Loading train.json...")
    # Load the json directly.
    with open(train_json_path, "r") as f:
        train_data = json.load(f)

    # Create DataFrame, excluding bands to save space in metadata
    # We only keep id, inc_angle, is_iceberg, and create a pointer to the file and index
    train_records = []
    for idx, item in enumerate(train_data):
        record = {
            "id": item["id"],
            "inc_angle": item["inc_angle"],
            "is_iceberg": item["is_iceberg"],
            "filepath": "train.json",
            "sample_index": idx,
        }
        train_records.append(record)

    df_train_full = pd.DataFrame(train_records)

    # Handle inc_angle 'na' values by coercing to NaN
    df_train_full["inc_angle"] = pd.to_numeric(
        df_train_full["inc_angle"], errors="coerce"
    )

    # Stratified Split
    print("Splitting training data...")
    X = df_train_full
    y = df_train_full["is_iceberg"]

    df_train, df_val = train_test_split(
        X, test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Save Train and Val Metadata
    train_meta_path = os.path.join(METADATA_DIR, "train_metadata.csv")
    val_meta_path = os.path.join(METADATA_DIR, "val_metadata.csv")

    df_train.to_csv(train_meta_path, index=False)
    df_val.to_csv(val_meta_path, index=False)
    print(f"Saved {train_meta_path}")
    print(f"Saved {val_meta_path}")

    # --- Process Test Data ---
    print("Loading test.json...")
    with open(test_json_path, "r") as f:
        test_data = json.load(f)

    test_records = []
    for idx, item in enumerate(test_data):
        record = {
            "id": item["id"],
            "inc_angle": item["inc_angle"],
            "filepath": "test.json",
            "sample_index": idx,
        }
        test_records.append(record)

    df_test = pd.DataFrame(test_records)
    df_test["inc_angle"] = pd.to_numeric(df_test["inc_angle"], errors="coerce")

    test_meta_path = os.path.join(METADATA_DIR, "test_metadata.csv")
    df_test.to_csv(test_meta_path, index=False)
    print(f"Saved {test_meta_path}")

    return df_train, df_val, df_test


def validate_metadata(df_train, df_val, df_test):
    print("\n--- Validating Metadata ---")

    # 1. Summary Statistics
    print(f"Train set shape: {df_train.shape}")
    print(f"Val set shape: {df_val.shape}")
    print(f"Test set shape: {df_test.shape}")

    print("\nClass Distribution (is_iceberg):")
    print(f"Train:\n{df_train['is_iceberg'].value_counts(normalize=True)}")
    print(f"Val:\n{df_val['is_iceberg'].value_counts(normalize=True)}")

    # 2. Check File Paths
    print("\nChecking file paths...")

    def check_paths(df, name):
        if "filepath" not in df.columns:
            return

        # Sample up to 1000 paths
        n_samples = min(1000, len(df))
        sample_paths = df["filepath"].sample(n=n_samples, random_state=RANDOM_STATE)

        missing_count = 0
        missing_samples = []

        for rel_path in sample_paths:
            full_path = os.path.join(INPUT_DIR, rel_path)
            if not os.path.exists(full_path):
                missing_count += 1
                if len(missing_samples) < 5:
                    missing_samples.append(rel_path)

        missing_ratio = missing_count / n_samples
        print(f"{name} missing file ratio: {missing_ratio:.4f}")

        if missing_ratio > 0.5:
            print(f"Sample missing paths: {missing_samples}")
            raise FileNotFoundError(
                f"More than 50% of file paths in {name} are invalid."
            )

    check_paths(df_train, "Train")
    check_paths(df_val, "Val")
    check_paths(df_test, "Test")

    # 3. Verify Split Requirements
    print("\nVerifying split requirements...")

    # Check Split Ratio
    total_train_val = len(df_train) + len(df_val)
    val_ratio = len(df_val) / total_train_val
    print(f"Actual Validation Ratio: {val_ratio:.4f}")

    # Allow small deviation due to integer division
    if not (0.19 < val_ratio < 0.21):
        raise AssertionError(
            f"Validation split ratio {val_ratio:.4f} is not close to 0.2"
        )

    # Check Stratification
    train_mean = df_train["is_iceberg"].mean()
    val_mean = df_val["is_iceberg"].mean()
    print(f"Train target mean: {train_mean:.4f}")
    print(f"Val target mean: {val_mean:.4f}")

    # Stratification check (tolerance 0.05)
    if abs(train_mean - val_mean) > 0.05:
        raise AssertionError(
            "Stratification failed: Class distributions differ significantly."
        )

    print("\nAll validation checks passed.")


if __name__ == "__main__":
    df_train, df_val, df_test = generate_metadata()
    validate_metadata(df_train, df_val, df_test)
