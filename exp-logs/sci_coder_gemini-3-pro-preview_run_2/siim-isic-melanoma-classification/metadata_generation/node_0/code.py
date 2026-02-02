import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def main():
    # Configuration
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    RANDOM_STATE = 42
    VAL_SPLIT_RATIO = 0.2
    # Number of splits = 1 / 0.2 = 5
    N_SPLITS = 5

    # 1. Setup Directories
    os.makedirs(METADATA_DIR, exist_ok=True)

    print("Loading raw data...")
    train_csv_path = os.path.join(INPUT_DIR, "train.csv")
    test_csv_path = os.path.join(INPUT_DIR, "test.csv")

    if not os.path.exists(train_csv_path) or not os.path.exists(test_csv_path):
        raise FileNotFoundError(
            "Input CSV files (train.csv/test.csv) not found in ./input"
        )

    df_train_full = pd.read_csv(train_csv_path)
    df_test = pd.read_csv(test_csv_path)

    # 2. Generate Relative File Paths
    # Images are located in ./input/jpeg/train/ and ./input/jpeg/test/
    # We store paths relative to ./input, e.g., "jpeg/train/ISIC_xxxx.jpg"
    print("Generating file paths...")
    df_train_full["file_path"] = df_train_full["image_name"].apply(
        lambda x: f"jpeg/train/{x}.jpg"
    )
    df_test["file_path"] = df_test["image_name"].apply(lambda x: f"jpeg/test/{x}.jpg")

    # 3. Create Validation Split
    # Requirements: 80:20 split, Random State 42, Group Sampling by patient_id, Stratified by target
    print("Splitting training data into Train/Val...")

    # Handle missing patient_ids if any (treat as unique groups)
    if df_train_full["patient_id"].isnull().any():
        print("Warning: Missing patient_ids detected. Filling with placeholders.")
        df_train_full["patient_id"] = df_train_full["patient_id"].fillna(
            "unknown_patient"
        )

    sgkf = StratifiedGroupKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )

    # We take the first fold as the validation set
    # StratifiedGroupKFold ensures groups (patients) don't overlap and tries to preserve class ratio
    split_generator = sgkf.split(
        df_train_full, df_train_full["target"], groups=df_train_full["patient_id"]
    )

    train_idx, val_idx = next(split_generator)

    df_train = df_train_full.iloc[train_idx].copy()
    df_val = df_train_full.iloc[val_idx].copy()

    # 4. Save Metadata
    print(f"Saving metadata to {METADATA_DIR}...")
    train_meta_path = os.path.join(METADATA_DIR, "train.csv")
    val_meta_path = os.path.join(METADATA_DIR, "val.csv")
    test_meta_path = os.path.join(METADATA_DIR, "test.csv")

    df_train.to_csv(train_meta_path, index=False)
    df_val.to_csv(val_meta_path, index=False)
    df_test.to_csv(test_meta_path, index=False)

    # 5. Verification
    print("\n" + "=" * 40)
    print("VERIFYING DATASETS")
    print("=" * 40)

    # Reload datasets to ensure integrity
    d_train = pd.read_csv(train_meta_path)
    d_val = pd.read_csv(val_meta_path)
    d_test = pd.read_csv(test_meta_path)

    datasets = [("Train", d_train), ("Validation", d_val), ("Test", d_test)]

    # 5a. Summary Statistics
    for name, df in datasets:
        print(f"\n[{name} Set]")
        print(f"Samples: {len(df)}")
        print(f"Columns: {list(df.columns)}")
        if "patient_id" in df.columns:
            print(f"Unique Patients: {df['patient_id'].nunique()}")
        if "target" in df.columns:
            print(
                f"Class Distribution: {df['target'].value_counts(normalize=True).to_dict()}"
            )
            print(f"Malignant Count: {df['target'].sum()}")

    # 5b. File Path Verification
    print("\nChecking file path validity (sampling 1000 paths per dataset)...")
    for name, df in datasets:
        sample_size = min(1000, len(df))
        sample = df.sample(n=sample_size, random_state=RANDOM_STATE)

        missing_count = 0
        missing_samples = []

        for _, row in sample.iterrows():
            # Construct full path from relative path
            full_path = os.path.join(INPUT_DIR, row["file_path"])
            if not os.path.exists(full_path):
                missing_count += 1
                if len(missing_samples) < 5:
                    missing_samples.append(row["file_path"])

        missing_ratio = missing_count / sample_size
        print(f"{name}: Missing Ratio = {missing_ratio:.4f}")

        if missing_ratio > 0.5:
            print(f"Sample missing paths in {name}: {missing_samples}")
            raise FileNotFoundError(
                f"Validation Failed: >50% of file paths in {name} are invalid."
            )

    # 5c. Split Verification
    print("\nVerifying Split Constraints...")

    # Check for patient overlap
    train_patients = set(d_train["patient_id"])
    val_patients = set(d_val["patient_id"])
    overlap = train_patients.intersection(val_patients)

    if overlap:
        raise AssertionError(
            f"Validation Failed: Found {len(overlap)} patients overlapping between Train and Validation sets (Data Leakage)."
        )
    else:
        print("Pass: No patient overlap.")

    # Check split ratio
    total_len = len(d_train) + len(d_val)
    val_actual_ratio = len(d_val) / total_len
    print(f"Actual Validation Ratio: {val_actual_ratio:.4f}")

    # Allow tolerance of +/- 0.05 due to group constraints
    if not (0.15 <= val_actual_ratio <= 0.25):
        raise AssertionError(
            f"Validation Failed: Split ratio {val_actual_ratio:.4f} deviates significantly from 0.20."
        )
    else:
        print("Pass: Split ratio is within acceptable range.")

    print("\nMetadata generation and verification completed successfully.")


if __name__ == "__main__":
    main()
