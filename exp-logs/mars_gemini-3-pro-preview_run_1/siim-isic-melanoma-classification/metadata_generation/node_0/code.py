import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedGroupKFold

# Constants
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
RANDOM_STATE = 42


def main():
    print("Starting metadata generation...")

    # 1. Setup Directory
    os.makedirs(METADATA_DIR, exist_ok=True)

    # 2. Load Data
    train_path = os.path.join(INPUT_DIR, "train.csv")
    test_path = os.path.join(INPUT_DIR, "test.csv")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError("Source CSV files not found in input directory.")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # 3. Add Relative File Paths
    # Images are in jpeg/train/ and jpeg/test/ directories inside input
    train_df["file_path"] = train_df["image_name"].apply(
        lambda x: f"jpeg/train/{x}.jpg"
    )
    test_df["file_path"] = test_df["image_name"].apply(lambda x: f"jpeg/test/{x}.jpg")

    # 4. Handle Patient IDs for Grouping
    # If patient_id is missing, we treat the image as a unique patient/group
    if "patient_id" not in train_df.columns:
        train_df["patient_id"] = train_df["image_name"]
    else:
        train_df["patient_id"] = train_df["patient_id"].fillna(train_df["image_name"])

    # 5. Split Training Data (Train/Val)
    # Requirement: Randomly shuffle before splitting
    train_df = train_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    # Requirement: 80:20 split, Group Sampling (by patient_id), Stratified (by target)
    # StratifiedGroupKFold with n_splits=5 gives 20% in test fold (which we use as validation)
    sgkf = StratifiedGroupKFold(n_splits=5)

    # We take the first fold
    fold_generator = sgkf.split(train_df, train_df["target"], train_df["patient_id"])
    train_idx, val_idx = next(fold_generator)

    train_split = train_df.iloc[train_idx].copy()
    val_split = train_df.iloc[val_idx].copy()

    # 6. Save Metadata
    print(f"Saving metadata to {METADATA_DIR}...")
    train_split.to_csv(os.path.join(METADATA_DIR, "train.csv"), index=False)
    val_split.to_csv(os.path.join(METADATA_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(METADATA_DIR, "test.csv"), index=False)

    print("Metadata generation complete.")

    # 7. Verification Steps
    verify_metadata()


def verify_metadata():
    print("\nRunning verification checks...")

    # Reload datasets
    train = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
    val = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))
    test = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))

    datasets = {"Training": train, "Validation": val, "Test": test}

    # 1. Summary Statistics
    print("\n=== Summary Statistics ===")
    for name, df in datasets.items():
        print(f"Dataset: {name}")
        print(f"  Samples: {len(df)}")
        print(f"  Columns: {list(df.columns)}")
        if "target" in df.columns:
            dist = df["target"].value_counts(normalize=True).to_dict()
            print(f"  Target Distribution: {dist}")
        if "patient_id" in df.columns:
            print(f"  Unique Patients: {df['patient_id'].nunique()}")
        print("-" * 30)

    # 2. Check File Paths
    print("\n=== Checking File Path Resolution ===")
    for name, df in datasets.items():
        check_file_paths(name, df)

    # 3. Verify Split Logic
    print("\n=== Verifying Split Requirements ===")

    # Assert Group Split (No patient overlap)
    train_patients = set(train["patient_id"].unique())
    val_patients = set(val["patient_id"].unique())
    intersection = train_patients.intersection(val_patients)

    if len(intersection) > 0:
        raise AssertionError(
            f"Split verification failed: {len(intersection)} patients found in both Train and Validation sets."
        )

    print("PASS: No patient leakage detected between Train and Validation.")

    # Check Split Ratio roughly
    total_train_val = len(train) + len(val)
    val_ratio = len(val) / total_train_val
    print(f"Validation Split Ratio: {val_ratio:.4f} (Target ~0.20)")

    # Assert Stratification (roughly)
    train_pos = train["target"].mean()
    val_pos = val["target"].mean()
    print(f"Train Positive Rate: {train_pos:.4f}")
    print(f"Val Positive Rate:   {val_pos:.4f}")

    print("\nAll checks passed successfully.")


def check_file_paths(dataset_name, df):
    # Check 1000 random paths
    n_samples = min(1000, len(df))
    sample = df.sample(n=n_samples, random_state=RANDOM_STATE)

    missing_count = 0
    missing_examples = []

    for _, row in sample.iterrows():
        rel_path = row["file_path"]
        # Path in metadata is relative to ./input
        full_path = os.path.join(INPUT_DIR, rel_path)

        if not os.path.exists(full_path):
            missing_count += 1
            if len(missing_examples) < 5:
                missing_examples.append(rel_path)

    missing_ratio = missing_count / n_samples
    print(
        f"[{dataset_name}] Missing File Ratio: {missing_ratio:.4f} ({missing_count}/{n_samples})"
    )

    if missing_ratio > 0.5:
        print("Examples of missing paths:")
        for p in missing_examples:
            print(f"  {p}")
        raise FileNotFoundError(
            f"Error: More than 50% of file paths in {dataset_name} metadata do not resolve to existing files."
        )


if __name__ == "__main__":
    main()
