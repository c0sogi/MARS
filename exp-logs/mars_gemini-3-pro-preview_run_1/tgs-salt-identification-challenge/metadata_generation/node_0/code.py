import pandas as pd
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split


def main():
    # -------------------------------------------------------------------------
    # 1. Configuration and Setup
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    RANDOM_STATE = 42

    os.makedirs(METADATA_DIR, exist_ok=True)

    print("Loading raw data...")
    train_csv_path = os.path.join(INPUT_DIR, "train.csv")
    depths_csv_path = os.path.join(INPUT_DIR, "depths.csv")
    sample_sub_path = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Load raw data
    df_train_raw = pd.read_csv(train_csv_path)
    df_depths = pd.read_csv(depths_csv_path)
    df_test_raw = pd.read_csv(sample_sub_path)

    # -------------------------------------------------------------------------
    # 2. Preprocessing and Merging
    # -------------------------------------------------------------------------
    # Merge depths
    df_train_merged = df_train_raw.merge(df_depths, on="id", how="left")
    df_test_merged = df_test_raw.merge(df_depths, on="id", how="left")

    # Calculate salt coverage for stratification
    def calculate_coverage(rle_string):
        if not isinstance(rle_string, str) or pd.isna(rle_string):
            return 0.0
        # RLE format: start length start length ...
        rle_numbers = [int(x) for x in rle_string.split()]
        # Sum of lengths (every second number)
        total_pixels = sum(rle_numbers[1::2])
        return total_pixels / (101 * 101)

    print("Calculating coverage for stratification...")
    df_train_merged["coverage"] = df_train_merged["rle_mask"].apply(calculate_coverage)

    # Create coverage classes for stratification (10 bins)
    # We map coverage 0-1 to integers 0-10
    def get_coverage_class(cov):
        if cov == 0:
            return 0
        return int(np.ceil(cov * 10))

    df_train_merged["coverage_class"] = df_train_merged["coverage"].apply(
        get_coverage_class
    )

    # -------------------------------------------------------------------------
    # 3. Stratified Split
    # -------------------------------------------------------------------------
    print("Splitting data...")
    X = df_train_merged
    y = df_train_merged["coverage_class"]

    train_df, val_df = train_test_split(
        X, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # -------------------------------------------------------------------------
    # 4. Path Generation
    # -------------------------------------------------------------------------
    def add_relative_paths(df, dataset_type):
        # dataset_type is 'train' (for both train and val splits) or 'test'
        # Images are at {dataset_type}/images/{id}.png
        # Masks are at {dataset_type}/masks/{id}.png (only for train source)

        # Note: The validation set comes from the original 'train' folder structure
        folder_name = "train" if dataset_type in ["train", "val"] else "test"

        df["image_path"] = df["id"].apply(lambda x: f"{folder_name}/images/{x}.png")

        if dataset_type in ["train", "val"]:
            df["mask_path"] = df["id"].apply(lambda x: f"{folder_name}/masks/{x}.png")
        else:
            df["mask_path"] = None

        return df

    train_df = add_relative_paths(train_df.copy(), "train")
    val_df = add_relative_paths(val_df.copy(), "val")
    test_df = add_relative_paths(df_test_merged.copy(), "test")

    # -------------------------------------------------------------------------
    # 5. Save Metadata
    # -------------------------------------------------------------------------
    print("Saving metadata...")
    train_save_path = os.path.join(METADATA_DIR, "train.csv")
    val_save_path = os.path.join(METADATA_DIR, "val.csv")
    test_save_path = os.path.join(METADATA_DIR, "test.csv")

    train_df.to_csv(train_save_path, index=False)
    val_df.to_csv(val_save_path, index=False)
    test_df.to_csv(test_save_path, index=False)

    # -------------------------------------------------------------------------
    # 6. Validation and Checks
    # -------------------------------------------------------------------------
    print("\n--- Performing Validation Checks ---")

    # Reload data
    final_train = pd.read_csv(train_save_path)
    final_val = pd.read_csv(val_save_path)
    final_test = pd.read_csv(test_save_path)

    # 6.1 Summary Statistics
    print(f"Train set shape: {final_train.shape}")
    print(f"Val set shape: {final_val.shape}")
    print(f"Test set shape: {final_test.shape}")

    print("\nTrain Coverage Class Distribution:")
    print(final_train["coverage_class"].value_counts(normalize=True).sort_index())
    print("\nVal Coverage Class Distribution:")
    print(final_val["coverage_class"].value_counts(normalize=True).sort_index())

    # 6.2 File Path Verification
    print("\nChecking random file paths...")
    all_dfs = [final_train, final_val, final_test]
    all_paths = []
    for df in all_dfs:
        if "image_path" in df.columns:
            all_paths.extend(df["image_path"].tolist())
        if "mask_path" in df.columns:
            # Filter out NaNs if any (test set has no mask paths, but column might exist as None/NaN)
            mask_paths = df["mask_path"].dropna().tolist()
            all_paths.extend(mask_paths)

    # Sample 1000 paths
    if len(all_paths) > 1000:
        sample_paths = np.random.choice(all_paths, 1000, replace=False)
    else:
        sample_paths = all_paths

    missing_count = 0
    missing_samples = []

    for rel_path in sample_paths:
        full_path = os.path.join(INPUT_DIR, rel_path)
        if not os.path.exists(full_path):
            missing_count += 1
            if len(missing_samples) < 5:
                missing_samples.append(rel_path)

    missing_ratio = missing_count / len(sample_paths)
    print(
        f"Checked {len(sample_paths)} paths. Missing: {missing_count}. Ratio: {missing_ratio:.4f}"
    )

    if missing_ratio > 0.5:
        print("Sample of missing paths:")
        for p in missing_samples:
            print(p)
        raise FileNotFoundError(
            f"Missing file ratio {missing_ratio} exceeds threshold of 0.5"
        )

    # 6.3 Stratification Verification
    print("\nVerifying stratification...")
    train_dist = final_train["coverage_class"].value_counts(normalize=True).sort_index()
    val_dist = final_val["coverage_class"].value_counts(normalize=True).sort_index()

    # Align indices to ensure we compare same classes (fill 0 for missing classes if any)
    all_classes = sorted(list(set(train_dist.index) | set(val_dist.index)))
    train_dist = train_dist.reindex(all_classes, fill_value=0)
    val_dist = val_dist.reindex(all_classes, fill_value=0)

    # Calculate max absolute difference in proportions
    diffs = np.abs(train_dist - val_dist)
    max_diff = diffs.max()

    print(
        f"Maximum difference in class proportions between Train and Val: {max_diff:.4f}"
    )

    # Threshold for stratification failure.
    # With small classes, slight variations happen, but shouldn't be massive.
    if max_diff > 0.05:
        raise AssertionError(
            f"Stratification failed. Max difference in class proportions: {max_diff}"
        )

    print("\nMetadata generation and validation completed successfully.")


if __name__ == "__main__":
    main()
