import pandas as pd
import numpy as np
import os
from sklearn.model_selection import GroupShuffleSplit


def main():
    # ==========================================
    # Configuration
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    RANDOM_STATE = 42
    VAL_SIZE = 0.2

    # Ensure metadata directory exists
    os.makedirs(METADATA_DIR, exist_ok=True)

    # ==========================================
    # 1. Load Data
    # ==========================================
    print("Loading raw data from input directory...")
    train_path = os.path.join(INPUT_DIR, "train.csv")
    test_path = os.path.join(INPUT_DIR, "test.csv")
    sample_sub_path = os.path.join(INPUT_DIR, "sample_submission.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    sample_sub = pd.read_csv(sample_sub_path)

    # Identify target columns (all columns in sample_submission except qa_id)
    target_cols = [col for col in sample_sub.columns if col != "qa_id"]
    print(f"Identified {len(target_cols)} target columns.")

    # ==========================================
    # 2. Split Data (Train/Val)
    # ==========================================
    # We use 'question_body' as the grouping key.
    # This ensures that all answers belonging to the same question end up in the same split,
    # preventing information leakage.
    group_col = "question_body"

    if group_col not in train_df.columns:
        # Fallback if specific column name differs, though unlikely for this dataset structure
        print(f"Warning: '{group_col}' not found. Falling back to 'qa_id'.")
        group_col = "qa_id"

    print(f"Splitting training data using GroupShuffleSplit on '{group_col}'...")
    print(f"Target Validation Size: {VAL_SIZE}, Random State: {RANDOM_STATE}")

    splitter = GroupShuffleSplit(
        n_splits=1, test_size=VAL_SIZE, random_state=RANDOM_STATE
    )

    # Get indices for the split
    groups = train_df[group_col]
    train_idx, val_idx = next(splitter.split(train_df, groups=groups))

    # Create DataFrames
    train_split = train_df.iloc[train_idx].copy()
    val_split = train_df.iloc[val_idx].copy()

    # ==========================================
    # 3. Generate Metadata
    # ==========================================
    print("Saving metadata files to ./metadata...")

    train_meta_path = os.path.join(METADATA_DIR, "train.csv")
    val_meta_path = os.path.join(METADATA_DIR, "val.csv")
    test_meta_path = os.path.join(METADATA_DIR, "test.csv")

    # We save the dataframes directly as metadata.
    # For text datasets of this size, this is efficient and allows direct loading.
    train_split.to_csv(train_meta_path, index=False)
    val_split.to_csv(val_meta_path, index=False)
    test_df.to_csv(test_meta_path, index=False)

    print("Metadata generation complete.")

    # ==========================================
    # 4. Verification & Checks
    # ==========================================
    print("\nPerforming verification checks...")

    # Reload data to verify integrity
    train_check = pd.read_csv(train_meta_path)
    val_check = pd.read_csv(val_meta_path)
    test_check = pd.read_csv(test_meta_path)

    # A. Summary Statistics
    print("\n--- Summary Statistics ---")
    print(f"Train Set: {train_check.shape[0]} rows, {train_check.shape[1]} columns")
    print(f"Val Set:   {val_check.shape[0]} rows, {val_check.shape[1]} columns")
    print(f"Test Set:  {test_check.shape[0]} rows, {test_check.shape[1]} columns")

    # Check if targets are present in train/val
    missing_targets = [t for t in target_cols if t not in train_check.columns]
    if missing_targets:
        raise AssertionError(
            f"Missing target columns in training metadata: {missing_targets}"
        )

    print("\nTarget Distribution (First 5 columns mean):")
    print(train_check[target_cols[:5]].mean())

    # B. Verify Group Split (No Leakage)
    print("\n--- Verifying Group Split ---")
    train_groups = set(train_check[group_col])
    val_groups = set(val_check[group_col])
    intersection = train_groups.intersection(val_groups)

    print(f"Unique groups in Train: {len(train_groups)}")
    print(f"Unique groups in Val:   {len(val_groups)}")
    print(f"Intersection size:      {len(intersection)}")

    if len(intersection) > 0:
        raise AssertionError(
            f"CRITICAL: Group leakage detected! {len(intersection)} groups found in both train and validation sets."
        )
    else:
        print("Success: No group leakage detected.")

    # C. Verify Split Ratio
    total_samples = len(train_check) + len(val_check)
    actual_val_ratio = len(val_check) / total_samples
    print(f"\nActual Validation Ratio: {actual_val_ratio:.4f}")

    # We allow a slightly larger tolerance because GroupShuffleSplit cannot split groups perfectly evenly
    if not (0.15 <= actual_val_ratio <= 0.25):
        raise AssertionError(
            f"Validation split ratio {actual_val_ratio:.4f} is outside the acceptable range [0.15, 0.25]."
        )
    else:
        print("Success: Split ratio is within acceptable range.")

    # D. File Path Check
    # The dataset consists of text within the CSVs, not references to external files.
    # Therefore, there are no relative file paths to check against ./input.
    print("\n--- File Path Verification ---")
    print(
        "No external file path columns detected (dataset is self-contained text). Skipping path resolution check."
    )

    print("\nAll verification checks passed.")


if __name__ == "__main__":
    main()
