import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import shutil


def generate_metadata():
    # Configuration
    INPUT_ROOT = "./input"
    METADATA_DIR = "./metadata"
    TRAIN_SUBDIR = "train2"
    TEST_SUBDIR = "test2"
    RANDOM_STATE = 42
    VAL_SIZE = 0.2

    # Ensure metadata directory exists
    if os.path.exists(METADATA_DIR):
        shutil.rmtree(METADATA_DIR)
    os.makedirs(METADATA_DIR)

    print("Scanning input directories...")

    # 1. Scan Training Data
    train_dir_path = os.path.join(INPUT_ROOT, TRAIN_SUBDIR)
    # Using glob to find .aif files. Assuming flat structure based on description.
    train_files = glob.glob(os.path.join(train_dir_path, "*.aif"))

    if not train_files:
        raise FileNotFoundError(f"No .aif files found in {train_dir_path}")

    train_data = []
    for filepath in train_files:
        filename = os.path.basename(filepath)
        # Requirement: "If the file ends in '_1.aif' it was labeled a whale call,
        # if it ends in '_0.aif', it was labeled noise."
        if filename.endswith("_1.aif"):
            label = 1
        elif filename.endswith("_0.aif"):
            label = 0
        else:
            # Fallback or error if naming convention is strict.
            # Based on description, all should match. We'll skip or mark as -1 if unexpected.
            # However, for this task, we assume the description covers the data.
            # Let's try to parse the last segment.
            try:
                # e.g. ...TRAIN0_0.aif -> 0
                label = int(filename.rsplit("_", 1)[-1].split(".")[0])
            except:
                print(f"Warning: Could not parse label for {filename}, skipping.")
                continue

        # Store path relative to ./input
        rel_path = os.path.join(TRAIN_SUBDIR, filename)
        train_data.append({"clip": filename, "filepath": rel_path, "label": label})

    df_full_train = pd.DataFrame(train_data)
    print(f"Found {len(df_full_train)} training samples.")

    # 2. Scan Test Data
    test_dir_path = os.path.join(INPUT_ROOT, TEST_SUBDIR)
    test_files = glob.glob(os.path.join(test_dir_path, "*.aif"))

    if not test_files:
        raise FileNotFoundError(f"No .aif files found in {test_dir_path}")

    test_data = []
    for filepath in test_files:
        filename = os.path.basename(filepath)
        rel_path = os.path.join(TEST_SUBDIR, filename)
        test_data.append({"clip": filename, "filepath": rel_path})  # No label for test

    df_test = pd.DataFrame(test_data)
    print(f"Found {len(df_test)} test samples.")

    # 3. Create Validation Split
    # Requirement: 80:20 split, stratified, random_state=42
    print("Splitting training data into train and validation sets...")

    X = df_full_train
    y = df_full_train["label"]

    df_train, df_val = train_test_split(
        X, test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=y, shuffle=True
    )

    # 4. Save Metadata
    print("Saving metadata to ./metadata ...")
    train_csv_path = os.path.join(METADATA_DIR, "train.csv")
    val_csv_path = os.path.join(METADATA_DIR, "val.csv")
    test_csv_path = os.path.join(METADATA_DIR, "test.csv")

    df_train.to_csv(train_csv_path, index=False)
    df_val.to_csv(val_csv_path, index=False)
    df_test.to_csv(test_csv_path, index=False)

    print("Metadata generation complete.")

    # ==========================================
    # Verification Steps
    # ==========================================
    print("\nStarting Verification Checks...")

    # Load datasets back
    df_train_loaded = pd.read_csv(train_csv_path)
    df_val_loaded = pd.read_csv(val_csv_path)
    df_test_loaded = pd.read_csv(test_csv_path)

    # Check 1: Summary Statistics
    print("\n--- Summary Statistics ---")
    print(f"Train Set: {len(df_train_loaded)} samples")
    print(
        f"Train Class Distribution:\n{df_train_loaded['label'].value_counts(normalize=True)}"
    )
    print(f"Val Set: {len(df_val_loaded)} samples")
    print(
        f"Val Class Distribution:\n{df_val_loaded['label'].value_counts(normalize=True)}"
    )
    print(f"Test Set: {len(df_test_loaded)} samples")

    # Check 2: File Path Resolution
    print("\n--- Checking File Paths ---")

    def check_paths(df, name):
        # Sample 1000 or all
        n_sample = min(1000, len(df))
        sample = df.sample(n=n_sample, random_state=RANDOM_STATE)

        missing_count = 0
        missing_examples = []

        for _, row in sample.iterrows():
            # Path is relative to ./input
            full_path = os.path.join(INPUT_ROOT, row["filepath"])
            if not os.path.exists(full_path):
                missing_count += 1
                if len(missing_examples) < 5:
                    missing_examples.append(full_path)

        ratio = missing_count / n_sample
        print(f"{name}: Checked {n_sample} paths. Missing ratio: {ratio:.4f}")

        if ratio > 0.5:
            print("Examples of missing paths:")
            for p in missing_examples:
                print(p)
            raise FileNotFoundError(
                f"More than 50% of file paths in {name} are invalid."
            )

    check_paths(df_train_loaded, "Train")
    check_paths(df_val_loaded, "Validation")
    check_paths(df_test_loaded, "Test")

    # Check 3: Validation Split Verification
    print("\n--- Verifying Split Logic ---")

    # Assert stratification
    train_pos_ratio = df_train_loaded["label"].mean()
    val_pos_ratio = df_val_loaded["label"].mean()

    print(f"Train Positive Ratio: {train_pos_ratio:.4f}")
    print(f"Val Positive Ratio:   {val_pos_ratio:.4f}")

    # Allow a small tolerance for stratification differences due to discrete counts
    if abs(train_pos_ratio - val_pos_ratio) > 0.05:
        raise AssertionError(
            "Stratification failed: Class distributions differ significantly between train and val."
        )

    # Assert no overlap
    train_ids = set(df_train_loaded["clip"])
    val_ids = set(df_val_loaded["clip"])
    overlap = train_ids.intersection(val_ids)

    if overlap:
        raise AssertionError(
            f"Data leakage detected: {len(overlap)} samples found in both train and val."
        )

    print("Split verification passed.")
    print("\nAll checks passed successfully.")


if __name__ == "__main__":
    generate_metadata()
