import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
import random

# Constants
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
TRAIN_DIR = os.path.join(INPUT_DIR, "train")
TEST_DIR = os.path.join(INPUT_DIR, "test")
RANDOM_STATE = 42


def get_file_dataframe(root_dir, split_name):
    """
    Walks through the directory structure to find all DICOM files.
    Structure: root_dir/study_id/series_id/image_id.dcm
    Returns a DataFrame with [study_id, series_id, image_id, file_path]
    """
    print(f"Scanning files in {root_dir}...")
    file_records = []

    # Using os.walk is generally faster/more robust than glob for deep trees
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".dcm"):
                # root is .../study_id/series_id
                # file is image_id.dcm

                # Get relative path from INPUT_DIR
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, INPUT_DIR)

                # Extract IDs from path
                # Path structure relative to root_dir: study/series/image.dcm
                # We need to be careful with path separators
                path_parts = os.path.relpath(abs_path, root_dir).split(os.sep)

                if len(path_parts) == 3:
                    study_id = path_parts[0]
                    series_id = path_parts[1]
                    image_id = os.path.splitext(path_parts[2])[0]

                    file_records.append(
                        {
                            "study_id": study_id,
                            "series_id": series_id,
                            "image_id": image_id,
                            "file_path": rel_path,
                            "dataset": split_name,
                        }
                    )

    df = pd.DataFrame(file_records)
    print(f"Found {len(df)} images in {split_name} set.")
    return df


def verify_file_paths(df, sample_size=1000):
    """
    Verifies that a sample of file paths in the dataframe actually exist.
    """
    if df.empty:
        return

    sample_n = min(len(df), sample_size)
    sample = df.sample(n=sample_n, random_state=RANDOM_STATE)

    missing_count = 0
    missing_samples = []

    for _, row in sample.iterrows():
        full_path = os.path.join(INPUT_DIR, row["file_path"])
        if not os.path.exists(full_path):
            missing_count += 1
            if len(missing_samples) < 5:
                missing_samples.append(row["file_path"])

    missing_ratio = missing_count / sample_n
    print(
        f"File existence check: {missing_count}/{sample_n} missing (Ratio: {missing_ratio:.4f})"
    )

    if missing_ratio > 0.5:
        print("Sample of missing files:")
        for p in missing_samples:
            print(f" - {p}")
        raise FileNotFoundError(
            f"Too many file paths do not resolve. Missing ratio: {missing_ratio}"
        )


def main():
    # 1. Setup
    os.makedirs(METADATA_DIR, exist_ok=True)

    # 2. Map Files
    train_files_df = get_file_dataframe(TRAIN_DIR, "train")
    test_files_df = get_file_dataframe(TEST_DIR, "test")

    # 3. Load Provided Metadata
    print("Loading CSV metadata...")
    study_level_csv = pd.read_csv(os.path.join(INPUT_DIR, "train_study_level.csv"))
    image_level_csv = pd.read_csv(os.path.join(INPUT_DIR, "train_image_level.csv"))

    # 4. Clean IDs
    # train_study_level.csv: id format "id_study"
    study_level_csv["study_id"] = study_level_csv["id"].str.replace(
        "_study", "", regex=False
    )
    study_level_csv = study_level_csv.rename(columns={"id": "study_level_id"})

    # train_image_level.csv: id format "id_image"
    image_level_csv["image_id"] = image_level_csv["id"].str.replace(
        "_image", "", regex=False
    )
    image_level_csv = image_level_csv.rename(columns={"id": "image_level_id"})

    # 5. Merge Data
    # Merge file info with image level info
    # We use inner join to ensure we only keep images that exist in both file system and metadata
    merged_df = pd.merge(train_files_df, image_level_csv, on="image_id", how="inner")

    # Merge with study level info
    merged_df = pd.merge(merged_df, study_level_csv, on="study_id", how="inner")

    print(f"Merged Data Shape: {merged_df.shape}")
    print(f"Unique Studies: {merged_df['study_id'].nunique()}")
    print(f"Unique Images: {merged_df['image_id'].nunique()}")

    # 6. Split Train/Val
    # We must use Group Sampling on study_id to prevent leakage
    splitter = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=RANDOM_STATE)

    # The splitter returns indices
    train_idx, val_idx = next(splitter.split(merged_df, groups=merged_df["study_id"]))

    train_df = merged_df.iloc[train_idx].copy()
    val_df = merged_df.iloc[val_idx].copy()

    train_df["split"] = "train"
    val_df["split"] = "val"

    # 7. Save Metadata
    print("Saving metadata to ./metadata/ ...")
    train_df.to_csv(os.path.join(METADATA_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(METADATA_DIR, "val.csv"), index=False)
    test_files_df.to_csv(os.path.join(METADATA_DIR, "test.csv"), index=False)

    # 8. Verification & Stats
    print("\n==== Verification & Statistics ====")

    # A. Stats
    print(
        f"Train Set: {len(train_df)} images, {train_df['study_id'].nunique()} studies"
    )
    print(f"Val Set:   {len(val_df)} images, {val_df['study_id'].nunique()} studies")
    print(f"Test Set:  {len(test_files_df)} images")

    # Class distribution (Study Level)
    labels = [
        "Negative for Pneumonia",
        "Typical Appearance",
        "Indeterminate Appearance",
        "Atypical Appearance",
    ]
    print("\nTrain Study Label Distribution:")
    # We need to drop duplicates by study_id to count study labels correctly
    train_studies = train_df.drop_duplicates("study_id")
    for label in labels:
        count = train_studies[label].sum()
        print(f"  {label}: {count} ({count/len(train_studies):.2%})")

    print("\nVal Study Label Distribution:")
    val_studies = val_df.drop_duplicates("study_id")
    for label in labels:
        count = val_studies[label].sum()
        print(f"  {label}: {count} ({count/len(val_studies):.2%})")

    # B. File Path Verification
    print("\nVerifying file paths...")
    verify_file_paths(train_df)
    verify_file_paths(val_df)
    verify_file_paths(test_files_df)

    # C. Split Verification
    print("\nVerifying split integrity...")
    train_study_ids = set(train_df["study_id"].unique())
    val_study_ids = set(val_df["study_id"].unique())

    intersection = train_study_ids.intersection(val_study_ids)
    overlap_count = len(intersection)

    print(f"Study ID Overlap: {overlap_count}")

    if overlap_count > 0:
        raise AssertionError(
            f"Data Leakage Detected! {overlap_count} studies are in both Train and Val sets."
        )

    # Check split ratio (approximate based on studies)
    total_studies = len(train_study_ids) + len(val_study_ids)
    val_ratio = len(val_study_ids) / total_studies
    print(f"Validation Split Ratio (by study): {val_ratio:.4f}")

    if not (0.15 < val_ratio < 0.25):
        # Note: GroupShuffleSplit is exact on groups, but if groups are small/large it might vary slightly.
        # Usually it's very close to 0.2.
        print(
            "Warning: Validation ratio deviates significantly from 0.2, but this may be due to group size variance."
        )

    print("\nMetadata generation complete. Success.")


if __name__ == "__main__":
    main()
