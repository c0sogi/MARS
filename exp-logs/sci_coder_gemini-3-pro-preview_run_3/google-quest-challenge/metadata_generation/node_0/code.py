import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit


def generate_metadata():
    # Define directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    os.makedirs(METADATA_DIR, exist_ok=True)

    # File paths
    train_path = os.path.join(INPUT_DIR, "train.csv")
    test_path = os.path.join(INPUT_DIR, "test.csv")
    sample_sub_path = os.path.join(INPUT_DIR, "sample_submission.csv")

    print("Loading raw data...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    sample_sub_df = pd.read_csv(sample_sub_path)

    # Identify target columns
    # The first column is usually qa_id, the rest are targets
    target_cols = [col for col in sample_sub_df.columns if col != "qa_id"]
    print(f"Identified {len(target_cols)} target labels.")

    # -------------------------------------------------------------------------
    # Split Training Data (Group Sampling)
    # -------------------------------------------------------------------------
    # We group by 'question_body' to ensure multiple answers for the same question
    # end up in the same split.

    group_col = "question_body"

    if group_col not in train_df.columns:
        # Fallback if question_body is missing (unlikely given description)
        print(f"Warning: '{group_col}' not found. Using random split.")
        from sklearn.model_selection import train_test_split

        train_split, val_split = train_test_split(
            train_df, test_size=0.2, random_state=42
        )
    else:
        print(f"Splitting data using GroupShuffleSplit on '{group_col}'...")
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        # We need to handle potential NaNs in grouping column by filling them with a placeholder
        groups = train_df[group_col].fillna("UNKNOWN_QUESTION")

        train_idx, val_idx = next(gss.split(train_df, groups=groups))

        train_split = train_df.iloc[train_idx].copy()
        val_split = train_df.iloc[val_idx].copy()

    # Add source file reference for path checking requirement
    train_split["original_file"] = "train.csv"
    val_split["original_file"] = "train.csv"
    test_df["original_file"] = "test.csv"

    # -------------------------------------------------------------------------
    # Save Metadata
    # -------------------------------------------------------------------------
    print("Saving metadata to ./metadata/ ...")
    train_meta_path = os.path.join(METADATA_DIR, "train_metadata.csv")
    val_meta_path = os.path.join(METADATA_DIR, "val_metadata.csv")
    test_meta_path = os.path.join(METADATA_DIR, "test_metadata.csv")

    train_split.to_csv(train_meta_path, index=False)
    val_split.to_csv(val_meta_path, index=False)
    test_df.to_csv(test_meta_path, index=False)

    # -------------------------------------------------------------------------
    # Verification and Checks
    # -------------------------------------------------------------------------
    print("\nPerforming verification checks...")

    # 1. Load datasets
    df_train_new = pd.read_csv(train_meta_path)
    df_val_new = pd.read_csv(val_meta_path)
    df_test_new = pd.read_csv(test_meta_path)

    # 2. Print Summary Statistics
    print("-" * 30)
    print("Summary Statistics:")
    print(f"Train set shape: {df_train_new.shape}")
    print(f"Val set shape:   {df_val_new.shape}")
    print(f"Test set shape:  {df_test_new.shape}")

    # Check class distributions for a few targets (mean value)
    print("\nTarget Mean Values (First 5 targets):")
    for col in target_cols[:5]:
        print(
            f"{col}: Train={df_train_new[col].mean():.4f}, Val={df_val_new[col].mean():.4f}"
        )
    print("-" * 30)

    # 3. Check File Paths
    # We check the 'original_file' column relative to ./input
    def check_paths(df, name):
        if "original_file" not in df.columns:
            return

        paths = df["original_file"].sample(n=min(1000, len(df)), random_state=42).values
        missing_count = 0
        missing_samples = []

        for p in paths:
            full_path = os.path.join(INPUT_DIR, p)
            if not os.path.exists(full_path):
                missing_count += 1
                if len(missing_samples) < 5:
                    missing_samples.append(p)

        ratio = missing_count / len(paths)
        print(f"Missing file ratio for {name}: {ratio:.4f}")

        if ratio > 0.5:
            print("Sample missing paths:", missing_samples)
            raise FileNotFoundError(
                f"More than 50% of file paths in {name} are invalid."
            )

    check_paths(df_train_new, "Train Metadata")
    check_paths(df_val_new, "Val Metadata")
    check_paths(df_test_new, "Test Metadata")

    # 4. Verify Validation Split Requirements
    print("\nVerifying split requirements...")

    # Check Ratio
    total_train_val = len(df_train_new) + len(df_val_new)
    val_ratio = len(df_val_new) / total_train_val
    print(f"Validation Ratio: {val_ratio:.4f}")

    # Allow small deviation due to group sizes
    if not (0.15 < val_ratio < 0.25):
        raise AssertionError(
            f"Validation split ratio {val_ratio:.4f} is too far from 0.2"
        )

    # Check Group Leakage
    if group_col in df_train_new.columns:
        train_groups = set(df_train_new[group_col].dropna().unique())
        val_groups = set(df_val_new[group_col].dropna().unique())

        intersection = train_groups.intersection(val_groups)
        print(f"Group Intersection Count: {len(intersection)}")

        if len(intersection) > 0:
            raise AssertionError(
                f"Data leakage detected! {len(intersection)} groups appear in both train and validation sets."
            )

    print("\nMetadata generation and verification completed successfully.")


if __name__ == "__main__":
    generate_metadata()
