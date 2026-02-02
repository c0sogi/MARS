import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit


def main():
    # 1. Setup
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    os.makedirs(METADATA_DIR, exist_ok=True)

    RANDOM_STATE = 42

    print("Loading raw data...")
    # Load training data
    train_orders_path = os.path.join(INPUT_DIR, "train_orders.csv")
    train_ancestors_path = os.path.join(INPUT_DIR, "train_ancestors.csv")

    df_orders = pd.read_csv(train_orders_path)
    df_ancestors = pd.read_csv(train_ancestors_path)

    # Load test data (ids from sample_submission)
    sample_submission_path = os.path.join(INPUT_DIR, "sample_submission.csv")
    df_sample_sub = pd.read_csv(sample_submission_path)

    # 2. Merge and Preprocess Training Data
    # Merge orders with ancestors to get grouping info
    # We use left join. If an ID is not in ancestors, we treat it as its own ancestor.
    df_train_full = df_orders.merge(df_ancestors, on="id", how="left")

    # Fill missing ancestor_id with id (assuming no parent/ancestor info means it's a standalone notebook)
    df_train_full["ancestor_id"] = df_train_full["ancestor_id"].fillna(
        df_train_full["id"]
    )

    # Create relative file paths
    df_train_full["file_path"] = "train/" + df_train_full["id"] + ".json"

    # 3. Split Train/Val
    print("Splitting data into Train and Validation sets...")
    gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=RANDOM_STATE)

    # We split based on ancestor_id to ensure no leakage of related notebooks
    groups = df_train_full["ancestor_id"]
    train_idx, val_idx = next(gss.split(df_train_full, groups=groups))

    df_train = df_train_full.iloc[train_idx].copy()
    df_val = df_train_full.iloc[val_idx].copy()

    # 4. Process Test Data
    df_test = df_sample_sub[["id"]].copy()
    df_test["file_path"] = "test/" + df_test["id"] + ".json"

    # 5. Save Metadata
    print("Saving metadata...")
    train_save_path = os.path.join(METADATA_DIR, "train.csv")
    val_save_path = os.path.join(METADATA_DIR, "val.csv")
    test_save_path = os.path.join(METADATA_DIR, "test.csv")

    df_train.to_csv(train_save_path, index=False)
    df_val.to_csv(val_save_path, index=False)
    df_test.to_csv(test_save_path, index=False)

    print(f"Saved train metadata to {train_save_path}")
    print(f"Saved val metadata to {val_save_path}")
    print(f"Saved test metadata to {test_save_path}")

    # 6. Validation and Checks
    print("\nPerforming validation checks...")

    # Load back data
    df_train_loaded = pd.read_csv(train_save_path)
    df_val_loaded = pd.read_csv(val_save_path)
    df_test_loaded = pd.read_csv(test_save_path)

    # 6a. Summary Statistics
    print("-" * 30)
    print("Summary Statistics:")
    print(f"Train samples: {len(df_train_loaded)}")
    print(f"Val samples:   {len(df_val_loaded)}")
    print(f"Test samples:  {len(df_test_loaded)}")

    print(f"Train unique ancestors: {df_train_loaded['ancestor_id'].nunique()}")
    print(f"Val unique ancestors:   {df_val_loaded['ancestor_id'].nunique()}")

    # 6b. Verify Split Logic (Group Separation)
    train_ancestors = set(df_train_loaded["ancestor_id"].unique())
    val_ancestors = set(df_val_loaded["ancestor_id"].unique())
    intersection = train_ancestors.intersection(val_ancestors)

    if len(intersection) > 0:
        raise AssertionError(
            f"Data Leakage detected! {len(intersection)} ancestors found in both train and val sets."
        )
    else:
        print("Verification Passed: No ancestor overlap between train and val.")

    # 6c. Check File Paths
    def check_filepaths(df, name):
        print(f"Checking file paths for {name}...")
        # Sample 1000 paths or all if less than 1000
        n_sample = min(1000, len(df))
        sample_paths = df["file_path"].sample(n=n_sample, random_state=RANDOM_STATE)

        missing_count = 0
        missing_samples = []

        for rel_path in sample_paths:
            full_path = os.path.join(INPUT_DIR, rel_path)
            if not os.path.exists(full_path):
                missing_count += 1
                if len(missing_samples) < 5:
                    missing_samples.append(rel_path)

        missing_ratio = missing_count / n_sample
        print(f"  Missing file ratio: {missing_ratio:.4f} ({missing_count}/{n_sample})")

        if missing_ratio > 0.5:
            print("  Sample missing files:", missing_samples)
            raise FileNotFoundError(
                f"More than 50% of files missing in {name} dataset."
            )
        else:
            print("  File path check passed.")

    check_filepaths(df_train_loaded, "Train")
    check_filepaths(df_val_loaded, "Validation")
    check_filepaths(df_test_loaded, "Test")

    print("\nAll checks passed successfully.")


if __name__ == "__main__":
    main()
