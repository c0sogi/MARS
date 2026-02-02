import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def main():
    # Configuration
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    RANDOM_STATE = 42
    VAL_SIZE = 0.2

    # Ensure metadata directory exists
    os.makedirs(METADATA_DIR, exist_ok=True)

    print("Loading raw data...")
    train_csv_path = os.path.join(INPUT_DIR, "train.csv")
    sample_sub_path = os.path.join(INPUT_DIR, "sample_submission.csv")

    if not os.path.exists(train_csv_path):
        raise FileNotFoundError(f"Could not find {train_csv_path}")
    if not os.path.exists(sample_sub_path):
        raise FileNotFoundError(f"Could not find {sample_sub_path}")

    df_full = pd.read_csv(train_csv_path)
    df_test = pd.read_csv(sample_sub_path)

    # Add relative file paths
    # Note: train images are in 'train_images/' and test images in 'test_images/'
    df_full["file_path"] = df_full["image"].apply(
        lambda x: os.path.join("train_images", x)
    )
    df_test["file_path"] = df_test["image"].apply(
        lambda x: os.path.join("test_images", x)
    )

    print(f"Total training samples: {len(df_full)}")
    print(f"Total test samples: {len(df_test)}")

    # Perform Stratified Split
    # We stratify based on the 'labels' column (treating the unique string combination as the class)
    # Handle rare classes that appear fewer than 2 times (cannot be stratified)
    label_counts = df_full["labels"].value_counts()
    rare_labels = label_counts[label_counts < 2].index.tolist()

    if rare_labels:
        print(
            f"Warning: Found {len(rare_labels)} rare label combinations with only 1 sample. These will be placed in the training set."
        )
        # Separate rare and common samples
        df_rare = df_full[df_full["labels"].isin(rare_labels)]
        df_common = df_full[~df_full["labels"].isin(rare_labels)]

        # Stratified split on common samples
        train_split, val_split = train_test_split(
            df_common,
            test_size=VAL_SIZE,
            stratify=df_common["labels"],
            random_state=RANDOM_STATE,
        )

        # Add rare samples back to training
        train_split = pd.concat([train_split, df_rare], axis=0)

        # Shuffle training set again to mix in the rare samples
        train_split = train_split.sample(frac=1, random_state=RANDOM_STATE).reset_index(
            drop=True
        )
        val_split = val_split.reset_index(drop=True)

    else:
        # Standard stratified split
        train_split, val_split = train_test_split(
            df_full,
            test_size=VAL_SIZE,
            stratify=df_full["labels"],
            random_state=RANDOM_STATE,
        )
        train_split = train_split.reset_index(drop=True)
        val_split = val_split.reset_index(drop=True)

    print(f"Split complete. Train: {len(train_split)}, Val: {len(val_split)}")

    # Save Metadata
    print("Saving metadata...")
    train_save_path = os.path.join(METADATA_DIR, "train.csv")
    val_save_path = os.path.join(METADATA_DIR, "val.csv")
    test_save_path = os.path.join(METADATA_DIR, "test.csv")

    train_split.to_csv(train_save_path, index=False)
    val_split.to_csv(val_save_path, index=False)
    df_test.to_csv(test_save_path, index=False)

    # ==========================================
    # Validation and Checks
    # ==========================================
    print("\nStarting validation checks...")

    # Reload datasets to ensure integrity
    df_train_check = pd.read_csv(train_save_path)
    df_val_check = pd.read_csv(val_save_path)
    df_test_check = pd.read_csv(test_save_path)

    # 1. Summary Statistics
    print("\n--- Summary Statistics ---")
    print(f"Train Shape: {df_train_check.shape}")
    print(f"Val Shape:   {df_val_check.shape}")
    print(f"Test Shape:  {df_test_check.shape}")

    print("\nTrain Label Distribution (Top 5):")
    print(df_train_check["labels"].value_counts(normalize=True).head())
    print("\nVal Label Distribution (Top 5):")
    print(df_val_check["labels"].value_counts(normalize=True).head())

    # 2. File Path Existence Check
    def check_files_exist(df, name):
        print(f"\nChecking file paths for {name} dataset...")
        sample_size = min(1000, len(df))
        sample = df.sample(n=sample_size, random_state=RANDOM_STATE)

        missing_count = 0
        missing_samples = []

        for _, row in sample.iterrows():
            rel_path = row["file_path"]
            # Resolve relative to INPUT_DIR
            full_path = os.path.join(INPUT_DIR, rel_path)
            if not os.path.exists(full_path):
                missing_count += 1
                if len(missing_samples) < 5:
                    missing_samples.append(rel_path)

        missing_ratio = missing_count / sample_size
        print(
            f"Missing File Ratio: {missing_ratio:.4f} ({missing_count}/{sample_size})"
        )

        if missing_ratio > 0.5:
            print("Sample missing paths:", missing_samples)
            raise FileNotFoundError(
                f"Error: More than 50% of file paths in {name} metadata do not resolve in {INPUT_DIR}."
            )

    check_files_exist(df_train_check, "Train")
    check_files_exist(df_val_check, "Validation")
    check_files_exist(df_test_check, "Test")

    # 3. Verify Split Requirements
    print("\nVerifying split requirements...")

    # Check split ratio (approximate, allowing for rounding and rare class handling)
    total_train_val = len(df_train_check) + len(df_val_check)
    actual_val_ratio = len(df_val_check) / total_train_val
    print(f"Actual Validation Ratio: {actual_val_ratio:.4f}")

    # Allow a small margin of error (e.g., +/- 1%) mostly due to rare class handling or small dataset size
    if not (0.19 <= actual_val_ratio <= 0.21):
        # If we had many rare classes forced to train, ratio might skew.
        # However, for this dataset, it should be close.
        print(
            "Warning: Validation ratio deviation detected (likely due to rare class handling)."
        )

    # Check Stratification
    # We compare the normalized value counts of the top classes
    train_dist = df_train_check["labels"].value_counts(normalize=True)
    val_dist = df_val_check["labels"].value_counts(normalize=True)

    # Align indices to compare
    all_labels = set(train_dist.index).union(set(val_dist.index))

    max_diff = 0.0
    for label in all_labels:
        t_freq = train_dist.get(label, 0)
        v_freq = val_dist.get(label, 0)
        diff = abs(t_freq - v_freq)
        if diff > max_diff:
            max_diff = diff

    print(
        f"Maximum difference in class probability between Train and Val: {max_diff:.4f}"
    )

    # Threshold for stratification failure:
    # If the max difference for any class proportion is > 0.05 (5%), we consider it a failure
    # (unless the class is very rare, but normalize handles proportions).
    # Given the dataset size, stratification should be tight.
    if max_diff > 0.05:
        raise AssertionError(
            "Stratification check failed: Distribution mismatch between train and validation sets is too high."
        )

    print("\nAll checks passed successfully.")


if __name__ == "__main__":
    main()
