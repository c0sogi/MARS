import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, GroupShuffleSplit

# Constants
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
RANDOM_STATE = 42
VAL_SIZE = 0.2


def main():
    # 1. Setup
    if not os.path.exists(METADATA_DIR):
        os.makedirs(METADATA_DIR)

    print("Starting metadata generation...")

    # 2. Load Data
    train_csv_path = os.path.join(INPUT_DIR, "train.csv")
    test_csv_path = os.path.join(INPUT_DIR, "test.csv")

    # Fallback for test.csv if it doesn't exist or is empty, try sample_submission
    if not os.path.exists(test_csv_path):
        test_csv_path = os.path.join(INPUT_DIR, "sample_submission.csv")

    df_train_full = pd.read_csv(train_csv_path)
    df_test = pd.read_csv(test_csv_path)

    # Standardize column names to match expected schema
    for df in [df_train_full, df_test]:
        if "Id" not in df.columns:
            if "id" in df.columns:
                df.rename(columns={"id": "Id"}, inplace=True)
            elif "image_id" in df.columns:
                df.rename(columns={"image_id": "Id"}, inplace=True)

    if (
        "Category" not in df_train_full.columns
        and "category_id" in df_train_full.columns
    ):
        df_train_full.rename(columns={"category_id": "Category"}, inplace=True)

    # 3. Construct File Paths
    # Assuming the image filename corresponds to the Id column with .jpg extension
    # train_images are in 'train_images/', test_images in 'test_images/'

    # Ensure Id column is string
    df_train_full["Id"] = df_train_full["Id"].astype(str)
    df_test["Id"] = df_test["Id"].astype(str)

    df_train_full["file_path"] = df_train_full["Id"].apply(
        lambda x: os.path.join("train_images", f"{x}.jpg")
    )
    df_test["file_path"] = df_test["Id"].apply(
        lambda x: os.path.join("test_images", f"{x}.jpg")
    )

    # 4. Split Training Data
    # Determine split strategy
    # The prompt mentions "138 different locations". If a 'location' or 'seq_id' column exists,
    # we should use Group Sampling. Otherwise, Stratified Sampling on 'Category'.

    # Check for potential group columns
    group_col = None
    possible_group_cols = ["location", "location_id", "seq_id", "sequence_id"]
    for col in possible_group_cols:
        if col in df_train_full.columns:
            group_col = col
            break

    if group_col:
        print(f"Group column '{group_col}' found. Using Group Sampling.")
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=VAL_SIZE, random_state=RANDOM_STATE
        )
        train_idx, val_idx = next(
            splitter.split(
                df_train_full,
                df_train_full["Category"],
                groups=df_train_full[group_col],
            )
        )
    else:
        print("No group column found. Using Stratified Sampling on 'Category'.")
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=VAL_SIZE, random_state=RANDOM_STATE
        )
        train_idx, val_idx = next(
            splitter.split(df_train_full, df_train_full["Category"])
        )

    df_train = df_train_full.iloc[train_idx].copy()
    df_val = df_train_full.iloc[val_idx].copy()

    # 5. Save Metadata
    train_meta_path = os.path.join(METADATA_DIR, "train_metadata.csv")
    val_meta_path = os.path.join(METADATA_DIR, "val_metadata.csv")
    test_meta_path = os.path.join(METADATA_DIR, "test_metadata.csv")

    df_train.to_csv(train_meta_path, index=False)
    df_val.to_csv(val_meta_path, index=False)
    df_test.to_csv(test_meta_path, index=False)

    print("Metadata files generated.")

    # 6. Verification
    verify_metadata(
        train_meta_path, val_meta_path, test_meta_path, df_train_full, group_col
    )


def verify_metadata(train_path, val_path, test_path, original_train_df, group_col):
    print("\n--- Verifying Metadata ---")

    # Load datasets
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    # Print Summary Statistics
    print(f"Train set shape: {df_train.shape}")
    print(f"Val set shape: {df_val.shape}")
    print(f"Test set shape: {df_test.shape}")

    print("\nTrain Class Distribution (Top 5):")
    print(df_train["Category"].value_counts(normalize=True).head())
    print("\nVal Class Distribution (Top 5):")
    print(df_val["Category"].value_counts(normalize=True).head())

    # Check File Paths
    for name, df in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
        check_file_paths(name, df)

    # Verify Split Requirements
    # 1. Ratio check
    total_train_val = len(df_train) + len(df_val)
    val_ratio = len(df_val) / total_train_val
    print(f"\nValidation Split Ratio: {val_ratio:.4f} (Target: {VAL_SIZE})")

    if not (0.19 < val_ratio < 0.21):
        raise AssertionError(
            f"Validation split ratio {val_ratio} deviates significantly from 0.2"
        )

    # 2. Stratification/Group Check
    if group_col:
        # Check no group leakage
        train_groups = set(df_train[group_col].unique())
        val_groups = set(df_val[group_col].unique())
        intersection = train_groups.intersection(val_groups)
        if intersection:
            raise AssertionError(
                f"Group leakage detected! {len(intersection)} groups overlap between train and val."
            )
        print("Group split verification passed: No group overlap.")
    else:
        # Check stratification consistency
        # We compare the distribution of categories.
        # For rare classes, exact match isn't possible, but overall distribution should be close.
        train_dist = df_train["Category"].value_counts(normalize=True).sort_index()
        val_dist = df_val["Category"].value_counts(normalize=True).sort_index()

        # Align indices
        all_cats = sorted(list(set(train_dist.index) | set(val_dist.index)))
        train_dist = train_dist.reindex(all_cats, fill_value=0)
        val_dist = val_dist.reindex(all_cats, fill_value=0)

        diff = (train_dist - val_dist).abs().mean()
        print(f"Average absolute difference in class probabilities: {diff:.6f}")

        if (
            diff > 0.05
        ):  # Allow some tolerance, especially for many classes/rare classes
            raise AssertionError(
                "Stratification failed: Class distributions differ significantly."
            )
        print("Stratification verification passed.")

    print("\nAll verification checks passed successfully.")


def check_file_paths(dataset_name, df):
    print(f"\nChecking file paths for {dataset_name}...")

    # Sample 1000 paths (or all if less than 1000)
    n_samples = min(1000, len(df))
    if n_samples == 0:
        print(f"No samples in {dataset_name} to check.")
        return

    sample_paths = df["file_path"].sample(n=n_samples, random_state=42).values

    missing_count = 0
    missing_samples = []

    for rel_path in sample_paths:
        full_path = os.path.join(INPUT_DIR, rel_path)
        if not os.path.exists(full_path):
            missing_count += 1
            if len(missing_samples) < 5:
                missing_samples.append(rel_path)

    missing_ratio = missing_count / n_samples
    print(f"Missing file ratio: {missing_ratio:.4f} ({missing_count}/{n_samples})")

    if missing_ratio > 0.5:
        print("Sample of missing paths:")
        for p in missing_samples:
            print(f" - {p}")
        raise FileNotFoundError(
            f"More than 50% of file paths in {dataset_name} are invalid."
        )


if __name__ == "__main__":
    main()
