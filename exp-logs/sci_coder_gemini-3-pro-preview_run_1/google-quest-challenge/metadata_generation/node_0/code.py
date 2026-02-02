import pandas as pd
import numpy as np
import os
from sklearn.model_selection import GroupShuffleSplit, train_test_split


def main():
    # Define paths
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    TRAIN_PATH = os.path.join(INPUT_DIR, "train.csv")
    TEST_PATH = os.path.join(INPUT_DIR, "test.csv")
    SAMPLE_SUB_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Create metadata directory
    os.makedirs(METADATA_DIR, exist_ok=True)

    # Load data
    print("Loading data...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

    # Identify target columns (excluding qa_id)
    target_cols = [col for col in sample_sub.columns if col != "qa_id"]
    print(f"Identified {len(target_cols)} target columns.")

    # Determine grouping strategy
    # We prefer grouping by question_body to avoid leakage of the same question into val
    group_col = None
    if "question_body" in train_df.columns:
        group_col = "question_body"
        print(f"Grouping by '{group_col}' for split.")
    elif "question_title" in train_df.columns:
        group_col = "question_title"
        print(f"Grouping by '{group_col}' for split.")
    else:
        print(
            "No specific question text column found for grouping. Using random split."
        )

    # Split data
    RANDOM_STATE = 42
    TEST_SIZE = 0.2

    if group_col:
        gss = GroupShuffleSplit(
            n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        train_idx, val_idx = next(gss.split(train_df, groups=train_df[group_col]))
        train_split = train_df.iloc[train_idx].copy()
        val_split = train_df.iloc[val_idx].copy()

        # Verification of groups
        train_groups = set(train_split[group_col].unique())
        val_groups = set(val_split[group_col].unique())
        intersection = train_groups.intersection(val_groups)
        if len(intersection) > 0:
            raise AssertionError(
                f"Group split failed. Found {len(intersection)} overlapping groups."
            )
        print(
            "Group split verification passed: No overlap between train and val groups."
        )

    else:
        # Fallback to random split if no grouping column found
        # We use stratified sampling based on a binned version of the first target if possible,
        # but given 30 continuous targets, simple random split is robust enough for this metadata generation task.
        train_split, val_split = train_test_split(
            train_df, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True
        )
        print("Random split performed.")

    # Add relative file paths
    # Paths are relative to ./input
    train_split["filepath"] = "train.csv"
    val_split["filepath"] = "train.csv"
    test_df["filepath"] = "test.csv"

    # Save metadata
    print("Saving metadata...")
    train_meta_path = os.path.join(METADATA_DIR, "train.csv")
    val_meta_path = os.path.join(METADATA_DIR, "val.csv")
    test_meta_path = os.path.join(METADATA_DIR, "test.csv")

    train_split.to_csv(train_meta_path, index=False)
    val_split.to_csv(val_meta_path, index=False)
    test_df.to_csv(test_meta_path, index=False)

    # ==========================================
    # Validation and Checks
    # ==========================================
    print("\nPerforming validation checks...")

    # Reload metadata
    meta_train = pd.read_csv(train_meta_path)
    meta_val = pd.read_csv(val_meta_path)
    meta_test = pd.read_csv(test_meta_path)

    # 1. Summary Statistics
    print("\n--- Summary Statistics ---")
    print(f"Train set shape: {meta_train.shape}")
    print(f"Val set shape:   {meta_val.shape}")
    print(f"Test set shape:  {meta_test.shape}")

    total_train_samples = len(train_df)
    print(f"Original Train samples: {total_train_samples}")
    print(
        f"Split Train samples: {len(meta_train)} ({len(meta_train)/total_train_samples:.2%})"
    )
    print(
        f"Split Val samples:   {len(meta_val)} ({len(meta_val)/total_train_samples:.2%})"
    )

    # Check target distribution (mean of means)
    if len(target_cols) > 0:
        train_target_mean = meta_train[target_cols].mean().mean()
        val_target_mean = meta_val[target_cols].mean().mean()
        print(f"Train global target mean: {train_target_mean:.4f}")
        print(f"Val global target mean:   {val_target_mean:.4f}")

    # 2. File Path Check
    print("\n--- Checking File Paths ---")

    def check_paths(df, name):
        if "filepath" not in df.columns:
            return

        # Sample 1000 paths
        sample_paths = (
            df["filepath"].sample(n=min(1000, len(df)), random_state=42).tolist()
        )
        missing_count = 0
        missing_samples = []

        for p in sample_paths:
            # Resolve path relative to input directory
            full_path = os.path.join(INPUT_DIR, p)
            if not os.path.exists(full_path):
                missing_count += 1
                if len(missing_samples) < 5:
                    missing_samples.append(p)

        missing_ratio = missing_count / len(sample_paths)
        print(f"[{name}] Missing file ratio: {missing_ratio:.4f}")

        if missing_ratio > 0.5:
            print(f"Sample missing paths: {missing_samples}")
            raise FileNotFoundError(
                f"More than 50% of file paths in {name} metadata do not resolve in {INPUT_DIR}."
            )

    check_paths(meta_train, "Train")
    check_paths(meta_val, "Val")
    check_paths(meta_test, "Test")

    # 3. Split Verification
    print("\n--- Verifying Split ---")
    # Verify ratio roughly 80:20
    actual_val_ratio = len(meta_val) / (len(meta_train) + len(meta_val))
    print(f"Actual Validation Ratio: {actual_val_ratio:.4f}")

    # Verify no id leakage
    train_ids = set(meta_train["qa_id"])
    val_ids = set(meta_val["qa_id"])
    id_intersection = train_ids.intersection(val_ids)
    if len(id_intersection) > 0:
        raise AssertionError(
            f"Found {len(id_intersection)} qa_ids overlapping between train and val."
        )

    # Verify group leakage if applicable
    if group_col:
        train_groups_check = set(meta_train[group_col].dropna().unique())
        val_groups_check = set(meta_val[group_col].dropna().unique())
        group_intersection = train_groups_check.intersection(val_groups_check)
        if len(group_intersection) > 0:
            raise AssertionError(
                f"Found {len(group_intersection)} groups overlapping between train and val."
            )
        print("Group split integrity verified.")

    print("\nMetadata generation and validation completed successfully.")


if __name__ == "__main__":
    main()
