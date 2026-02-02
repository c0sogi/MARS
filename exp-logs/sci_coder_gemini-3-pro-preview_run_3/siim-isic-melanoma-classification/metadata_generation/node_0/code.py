import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedGroupKFold

# Configuration
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
RANDOM_STATE = 42
VAL_SIZE = 0.2


def generate_metadata():
    """Generates train, validation, and test metadata CSVs."""
    print("Starting metadata generation...")

    # Create metadata directory
    os.makedirs(METADATA_DIR, exist_ok=True)

    # 1. Load Raw Data
    train_csv_path = os.path.join(INPUT_DIR, "train.csv")
    test_csv_path = os.path.join(INPUT_DIR, "test.csv")

    if not os.path.exists(train_csv_path) or not os.path.exists(test_csv_path):
        raise FileNotFoundError("Could not find train.csv or test.csv in ./input")

    df_train_full = pd.read_csv(train_csv_path)
    df_test = pd.read_csv(test_csv_path)

    print(
        f"Loaded {len(df_train_full)} training samples and {len(df_test)} test samples."
    )

    # 2. Construct Relative File Paths
    # Images are in jpeg/train/ and jpeg/test/
    df_train_full["file_path"] = df_train_full["image_name"].apply(
        lambda x: f"jpeg/train/{x}.jpg"
    )
    df_test["file_path"] = df_test["image_name"].apply(lambda x: f"jpeg/test/{x}.jpg")

    # 3. Create Validation Split
    # Requirement: 80:20 split, Random Shuffle, Group Sampling (by patient_id)

    # Handle missing patient_ids if any (treat as unique groups)
    if df_train_full["patient_id"].isnull().any():
        print("Note: Missing patient_ids detected. Filling with placeholder.")
        df_train_full["patient_id"] = df_train_full["patient_id"].fillna(
            "unknown_patient"
        )

    # Shuffle the data before splitting as per requirements
    df_train_full = df_train_full.sample(frac=1, random_state=RANDOM_STATE).reset_index(
        drop=True
    )

    X = df_train_full
    y = df_train_full["target"]
    groups = df_train_full["patient_id"]

    # Use StratifiedGroupKFold with n_splits=5 to get a ~20% validation set
    # This handles both stratification (class balance) and grouping (no patient leakage)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # We take the first fold as the validation set
    train_idx, val_idx = next(sgkf.split(X, y, groups))

    df_train = df_train_full.iloc[train_idx].copy()
    df_val = df_train_full.iloc[val_idx].copy()

    # 4. Save Metadata
    train_save_path = os.path.join(METADATA_DIR, "train.csv")
    val_save_path = os.path.join(METADATA_DIR, "val.csv")
    test_save_path = os.path.join(METADATA_DIR, "test.csv")

    df_train.to_csv(train_save_path, index=False)
    df_val.to_csv(val_save_path, index=False)
    df_test.to_csv(test_save_path, index=False)

    print(f"Saved metadata to {METADATA_DIR}")
    return train_save_path, val_save_path, test_save_path


def validate_metadata(train_path, val_path, test_path):
    """Performs validation checks on the generated metadata."""
    print("\n--- Validating Generated Metadata ---")

    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    # 1. Summary Statistics
    print("\n1. Summary Statistics:")
    print(
        f"  Train Set: {len(df_train)} samples, Target Mean: {df_train['target'].mean():.4f}"
    )
    print(
        f"  Val Set:   {len(df_val)} samples, Target Mean: {df_val['target'].mean():.4f}"
    )
    print(f"  Test Set:  {len(df_test)} samples")
    print(f"  Train Patients: {df_train['patient_id'].nunique()}")
    print(f"  Val Patients:   {df_val['patient_id'].nunique()}")

    # Check split ratio roughly
    total_train_val = len(df_train) + len(df_val)
    val_ratio = len(df_val) / total_train_val
    print(f"  Actual Validation Ratio: {val_ratio:.4f} (Target: ~0.2)")

    # 2. Check File Existence
    print("\n2. Checking File Paths (Sample 1000)...")

    def check_paths(df, name):
        paths = df["file_path"].values
        sample_size = min(1000, len(paths))
        # Use fixed seed for reproducibility of check
        np.random.seed(42)
        sampled_paths = np.random.choice(paths, sample_size, replace=False)

        missing_count = 0
        missing_examples = []

        for p in sampled_paths:
            # Paths are relative to ./input
            full_path = os.path.join(INPUT_DIR, p)
            if not os.path.exists(full_path):
                missing_count += 1
                if len(missing_examples) < 5:
                    missing_examples.append(p)

        ratio = missing_count / sample_size
        print(f"  [{name}] Missing Ratio: {ratio:.4f} ({missing_count}/{sample_size})")

        if ratio > 0.5:
            print(f"  [{name}] Example missing paths: {missing_examples}")
            raise FileNotFoundError(
                f"Error: More than 50% of file paths in {name} set are missing."
            )

    check_paths(df_train, "Train")
    check_paths(df_val, "Val")
    check_paths(df_test, "Test")

    # 3. Verify Split Logic (Group Leakage)
    print("\n3. Verifying Group Split Logic...")
    train_patients = set(df_train["patient_id"].unique())
    val_patients = set(df_val["patient_id"].unique())

    intersection = train_patients.intersection(val_patients)
    overlap_count = len(intersection)

    print(f"  Patient Overlap Count: {overlap_count}")

    if overlap_count > 0:
        raise AssertionError(
            f"CRITICAL: Data leakage detected. {overlap_count} patients appear in both train and validation sets."
        )

    # Verify Stratification (loose check, just to ensure both classes exist)
    if df_val["target"].nunique() < 2:
        print("  Warning: Validation set does not contain both classes.")
    else:
        print("  Stratification check: Both classes present in validation set.")

    print("\nAll validation checks passed successfully.")


if __name__ == "__main__":
    try:
        t_path, v_path, te_path = generate_metadata()
        validate_metadata(t_path, v_path, te_path)
    except Exception as e:
        print(f"\nExecution failed with error: {e}")
        raise e
