import os
import pandas as pd
import numpy as np
import random


def generate_metadata():
    # Define paths
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Create metadata directory
    os.makedirs(METADATA_DIR, exist_ok=True)

    # Load raw CSVs
    print("Loading raw data...")
    train_orders_path = os.path.join(INPUT_DIR, "train_orders.csv")
    train_ancestors_path = os.path.join(INPUT_DIR, "train_ancestors.csv")
    sample_submission_path = os.path.join(INPUT_DIR, "sample_submission.csv")

    df_orders = pd.read_csv(train_orders_path)
    df_ancestors = pd.read_csv(train_ancestors_path)
    df_test_sample = pd.read_csv(sample_submission_path)

    # --- Process Training Data ---

    # Merge orders with ancestors
    # We use left join because we need all training notebooks.
    # If ancestor info is missing, we treat the notebook as its own ancestor.
    df_train_full = df_orders.merge(df_ancestors, on="id", how="left")

    # Fill missing ancestor_ids with the notebook id itself
    df_train_full["ancestor_id"] = df_train_full["ancestor_id"].fillna(
        df_train_full["id"]
    )

    # Add relative file path
    df_train_full["filepath"] = df_train_full["id"].apply(lambda x: f"train/{x}.json")

    # Split into Train and Validation (Group Split)
    # We must split by ancestor_id to avoid data leakage (forks of the same notebook should be in the same set)
    unique_ancestors = df_train_full["ancestor_id"].unique()

    # Shuffle ancestors
    rng = np.random.default_rng(seed=42)
    rng.shuffle(unique_ancestors)

    # 80/20 Split
    split_idx = int(len(unique_ancestors) * 0.8)
    train_ancestors = set(unique_ancestors[:split_idx])
    val_ancestors = set(unique_ancestors[split_idx:])

    df_train = df_train_full[df_train_full["ancestor_id"].isin(train_ancestors)].copy()
    df_val = df_train_full[df_train_full["ancestor_id"].isin(val_ancestors)].copy()

    # --- Process Test Data ---

    df_test = df_test_sample[["id"]].copy()
    df_test["filepath"] = df_test["id"].apply(lambda x: f"test/{x}.json")
    # Test set doesn't have ancestor info or cell_order in the context of input data availability for inference,
    # but we include id and filepath.

    # --- Save Metadata ---
    print("Saving metadata...")
    train_meta_path = os.path.join(METADATA_DIR, "train_metadata.csv")
    val_meta_path = os.path.join(METADATA_DIR, "val_metadata.csv")
    test_meta_path = os.path.join(METADATA_DIR, "test_metadata.csv")

    df_train.to_csv(train_meta_path, index=False)
    df_val.to_csv(val_meta_path, index=False)
    df_test.to_csv(test_meta_path, index=False)

    print(f"Metadata saved to {METADATA_DIR}")

    # --- Verification ---
    print("Verifying metadata...")

    # 1. Load datasets
    df_train_loaded = pd.read_csv(train_meta_path)
    df_val_loaded = pd.read_csv(val_meta_path)
    df_test_loaded = pd.read_csv(test_meta_path)

    # 2. Print summary statistics
    print("\nSummary Statistics:")
    print(f"Train samples: {len(df_train_loaded)}")
    print(f"Val samples: {len(df_val_loaded)}")
    print(f"Test samples: {len(df_test_loaded)}")

    print(f"Train unique ancestors: {df_train_loaded['ancestor_id'].nunique()}")
    print(f"Val unique ancestors: {df_val_loaded['ancestor_id'].nunique()}")

    # 3. Check file paths
    def check_filepaths(df, name):
        paths = df["filepath"].tolist()
        if len(paths) > 1000:
            paths = random.sample(paths, 1000)

        missing_count = 0
        missing_samples = []

        for p in paths:
            full_path = os.path.join(INPUT_DIR, p)
            if not os.path.exists(full_path):
                missing_count += 1
                if len(missing_samples) < 5:
                    missing_samples.append(p)

        missing_ratio = missing_count / len(paths) if paths else 0
        print(f"Missing file ratio for {name}: {missing_ratio:.4f}")

        if missing_ratio > 0.5:
            print(f"Sample missing files for {name}: {missing_samples}")
            raise FileNotFoundError(f"Too many missing files in {name} dataset.")

    check_filepaths(df_train_loaded, "Train")
    check_filepaths(df_val_loaded, "Validation")
    check_filepaths(df_test_loaded, "Test")

    # 4. Verify Split Requirements
    # Assert stratification/group split success
    train_ancestors_set = set(df_train_loaded["ancestor_id"].unique())
    val_ancestors_set = set(df_val_loaded["ancestor_id"].unique())

    intersection = train_ancestors_set.intersection(val_ancestors_set)
    if intersection:
        raise AssertionError(
            f"Data leakage detected! {len(intersection)} ancestors found in both train and validation sets."
        )

    # Check split ratio roughly
    total_ancestors = len(train_ancestors_set) + len(val_ancestors_set)
    train_ratio = len(train_ancestors_set) / total_ancestors
    print(f"Actual split ratio (by ancestor groups): {train_ratio:.4f} (Target: 0.8)")

    # Allow small deviation due to discrete counts, but it should be close
    if not (0.78 < train_ratio < 0.82):
        print(
            "Warning: Split ratio deviates slightly from 0.8 due to grouping, which is expected."
        )

    print("\nAll verification checks passed successfully.")


if __name__ == "__main__":
    generate_metadata()
