import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split


def main():
    # ==========================================
    # Configuration & Setup
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    RANDOM_STATE = 42
    VAL_SIZE = 0.2

    # Create metadata directory
    os.makedirs(METADATA_DIR, exist_ok=True)

    print("Loading raw data from input directory...")
    try:
        train_df = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))
        test_df = pd.read_csv(os.path.join(INPUT_DIR, "test.csv"))
        sample_submission = pd.read_csv(
            os.path.join(INPUT_DIR, "sample_submission.csv")
        )
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return

    # ==========================================
    # Process Training Data (Stratified Split)
    # ==========================================
    print("Processing training data and performing stratified split...")

    label_cols = [
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate",
    ]

    # Create a stratification group based on the unique combination of labels
    # This ensures that the distribution of multi-label combinations is preserved
    train_df["stratify_group"] = train_df[label_cols].astype(str).agg("".join, axis=1)

    # Handle rare groups (count < 2) which cannot be stratified
    # We treat them as a separate 'rare' group for stratification purposes
    group_counts = train_df["stratify_group"].value_counts()
    rare_groups = group_counts[group_counts < 2].index
    train_df["stratify_safe"] = train_df["stratify_group"]
    train_df.loc[train_df["stratify_group"].isin(rare_groups), "stratify_safe"] = "rare"

    # Perform 80:20 Stratified Split
    train_split, val_split = train_test_split(
        train_df,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=train_df["stratify_safe"],
    )

    # Create Metadata DataFrames
    # We store ID, Labels, and the relative path to the source file.
    # We do NOT store the text content to avoid copying raw data.
    train_meta = train_split[["id"] + label_cols].copy()
    train_meta["source_file"] = "train.csv"

    val_meta = val_split[["id"] + label_cols].copy()
    val_meta["source_file"] = "train.csv"

    # ==========================================
    # Process Test Data
    # ==========================================
    print("Processing test data...")

    # The test set metadata should correspond to the IDs in sample_submission
    # We use sample_submission as the base to ensure correct IDs and order
    test_meta = sample_submission[["id"]].copy()
    test_meta["source_file"] = "test.csv"

    # ==========================================
    # Save Metadata
    # ==========================================
    print(f"Saving metadata to {METADATA_DIR}...")
    train_meta.to_csv(os.path.join(METADATA_DIR, "train.csv"), index=False)
    val_meta.to_csv(os.path.join(METADATA_DIR, "val.csv"), index=False)
    test_meta.to_csv(os.path.join(METADATA_DIR, "test.csv"), index=False)

    print("Metadata generation complete.")

    # ==========================================
    # Verification & Validation
    # ==========================================
    print("\nStarting Verification Checks...")

    def load_dataset_from_metadata(meta_filename):
        """
        Loads the full dataset by reading metadata and merging with raw source files.
        """
        meta_path = os.path.join(METADATA_DIR, meta_filename)
        meta_df = pd.read_csv(meta_path)

        # Identify sources
        sources = meta_df["source_file"].unique()
        loaded_data = []

        for src in sources:
            # Read raw source file
            src_path = os.path.join(INPUT_DIR, src)
            if not os.path.exists(src_path):
                raise FileNotFoundError(f"Source file {src} missing.")

            src_df = pd.read_csv(src_path)

            # Filter metadata for this source
            subset_meta = meta_df[meta_df["source_file"] == src]

            # Merge to get text content
            # We use inner join on ID
            merged = pd.merge(subset_meta, src_df, on="id", suffixes=("", "_src"))

            # Cleanup duplicate columns from merge (keep metadata labels if present)
            cols_to_drop = [c for c in merged.columns if c.endswith("_src")]
            merged = merged.drop(columns=cols_to_drop)

            loaded_data.append(merged)

        return pd.concat(loaded_data, ignore_index=True)

    # 1. Load Datasets using new metadata
    print("Loading datasets via metadata to verify integrity...")
    df_train_final = load_dataset_from_metadata("train.csv")
    df_val_final = load_dataset_from_metadata("val.csv")
    df_test_final = load_dataset_from_metadata("test.csv")

    # 2. Print Summary Statistics
    print("\n=== Dataset Summary ===")
    print(f"Train Set: {df_train_final.shape[0]} samples")
    print(f"Val Set:   {df_val_final.shape[0]} samples")
    print(f"Test Set:  {df_test_final.shape[0]} samples")

    print("\nClass Distribution (Train):")
    print(df_train_final[label_cols].mean())
    print("\nClass Distribution (Val):")
    print(df_val_final[label_cols].mean())

    # 3. Check File Paths
    print("\nChecking file path validity (sample of 1000)...")
    for meta_file in ["train.csv", "val.csv", "test.csv"]:
        meta_df = pd.read_csv(os.path.join(METADATA_DIR, meta_file))

        # Sample 1000 paths (or all if less than 1000)
        sample_size = min(1000, len(meta_df))
        sample_paths = meta_df["source_file"].sample(n=sample_size, random_state=42)

        missing_count = 0
        missing_examples = []

        for rel_path in sample_paths:
            full_path = os.path.join(INPUT_DIR, rel_path)
            if not os.path.exists(full_path):
                missing_count += 1
                if len(missing_examples) < 5:
                    missing_examples.append(rel_path)

        missing_ratio = missing_count / sample_size
        print(f"  {meta_file}: Missing Ratio = {missing_ratio:.4f}")

        if missing_ratio > 0.5:
            print(f"  Sample missing paths: {missing_examples}")
            raise FileNotFoundError(
                f"Validation failed: >50% of file paths in {meta_file} are invalid."
            )

    # 4. Verify Validation Split Requirements
    print("\nVerifying split requirements...")

    # Check Ratio
    total_train_val = len(df_train_final) + len(df_val_final)
    actual_val_ratio = len(df_val_final) / total_train_val
    print(f"  Target Val Ratio: {VAL_SIZE}")
    print(f"  Actual Val Ratio: {actual_val_ratio:.4f}")

    if abs(actual_val_ratio - VAL_SIZE) > 0.01:
        raise AssertionError(
            f"Validation split ratio {actual_val_ratio:.4f} deviates significantly from {VAL_SIZE}"
        )

    # Check Stratification (using 'toxic' label as proxy for distribution match)
    train_toxic_mean = df_train_final["toxic"].mean()
    val_toxic_mean = df_val_final["toxic"].mean()
    print(f"  Train 'toxic' mean: {train_toxic_mean:.4f}")
    print(f"  Val 'toxic' mean:   {val_toxic_mean:.4f}")

    # Allow a small margin of error for stratification
    if abs(train_toxic_mean - val_toxic_mean) > 0.01:
        raise AssertionError(
            "Stratified sampling failed: Class distributions differ significantly."
        )

    print("\nAll validation checks passed successfully!")


if __name__ == "__main__":
    main()
