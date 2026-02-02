import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Constants
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
RANDOM_STATE = 42


def get_file_dataframe(root_dir, split_name):
    """
    Walks the directory to find all .dcm files and extracts IDs.
    Returns a DataFrame with study_id, series_id, image_id, and relative file_path.
    """
    data = []
    # Pattern: root_dir/study_id/series_id/image_id.dcm
    # We use os.walk for efficiency over glob with wildcards for large directories
    abs_root = os.path.abspath(root_dir)

    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.endswith(".dcm"):
                # Get relative path from INPUT_DIR
                # dirpath is like ./input/train/study/series
                # rel_dir is like train/study/series
                rel_path = os.path.relpath(os.path.join(dirpath, f), INPUT_DIR)

                # Extract IDs from path parts
                # Path parts relative to the specific split folder (train or test)
                # Structure: split_name/study/series/image.dcm
                parts = rel_path.split(os.sep)

                # Ensure structure is as expected: [split_name, study, series, image.dcm]
                if len(parts) >= 4:
                    study_id = parts[-3]
                    series_id = parts[-2]
                    image_id = parts[-1].replace(".dcm", "")

                    data.append(
                        {
                            "study_id": study_id,
                            "series_id": series_id,
                            "image_id": image_id,
                            "file_path": rel_path,
                        }
                    )

    return pd.DataFrame(data)


def main():
    # 1. Setup
    if not os.path.exists(METADATA_DIR):
        os.makedirs(METADATA_DIR)

    print("Scanning file system...")
    # 2. Parse Filesystem
    df_train_files = get_file_dataframe(os.path.join(INPUT_DIR, "train"), "train")
    df_test_files = get_file_dataframe(os.path.join(INPUT_DIR, "test"), "test")

    print(
        f"Found {len(df_train_files)} training files and {len(df_test_files)} test files."
    )

    # 3. Load Provided Metadata
    train_study_df = pd.read_csv(os.path.join(INPUT_DIR, "train_study_level.csv"))
    train_image_df = pd.read_csv(os.path.join(INPUT_DIR, "train_image_level.csv"))

    # 4. Preprocess IDs
    # train_study_level: id is like '000c9c05fd14_study'
    train_study_df["study_id"] = train_study_df["id"].str.replace("_study", "")
    train_study_df = train_study_df.rename(columns={"id": "id_study_original"})

    # train_image_level: id is like '000c9c05fd14_image'
    train_image_df["image_id"] = train_image_df["id"].str.replace("_image", "")
    train_image_df = train_image_df.rename(columns={"id": "id_image_original"})

    # 5. Merge Data
    # Merge file info with image labels
    df_merged = pd.merge(df_train_files, train_image_df, on="image_id", how="inner")

    # Merge with study labels
    df_merged = pd.merge(df_merged, train_study_df, on="study_id", how="inner")

    print(f"Merged DataFrame shape: {df_merged.shape}")

    # 6. Create Validation Split (Group Sampling by Study)
    # We need to split based on study_id to avoid leakage
    unique_studies = df_merged[
        [
            "study_id",
            "Negative for Pneumonia",
            "Typical Appearance",
            "Indeterminate Appearance",
            "Atypical Appearance",
        ]
    ].drop_duplicates()

    # Create a stratification label for the studies
    # Since these are one-hot, we can convert to a single label for stratification purposes
    # We use argmax to get the index of the active class (0-3)
    label_cols = [
        "Negative for Pneumonia",
        "Typical Appearance",
        "Indeterminate Appearance",
        "Atypical Appearance",
    ]
    unique_studies["stratify_label"] = unique_studies[label_cols].values.argmax(axis=1)

    train_studies, val_studies = train_test_split(
        unique_studies["study_id"],
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=unique_studies["stratify_label"],
    )

    df_train = df_merged[df_merged["study_id"].isin(train_studies)].copy()
    df_val = df_merged[df_merged["study_id"].isin(val_studies)].copy()

    # 7. Save Metadata
    train_save_path = os.path.join(METADATA_DIR, "train.csv")
    val_save_path = os.path.join(METADATA_DIR, "val.csv")
    test_save_path = os.path.join(METADATA_DIR, "test.csv")

    df_train.to_csv(train_save_path, index=False)
    df_val.to_csv(val_save_path, index=False)
    df_test_files.to_csv(test_save_path, index=False)

    print("Metadata files saved.")

    # 8. Verification
    verify_metadata(train_save_path, val_save_path, test_save_path)


def verify_metadata(train_path, val_path, test_path):
    print("\n--- Verifying Metadata ---")

    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    # 1. Summary Statistics
    print(f"Train samples: {len(df_train)}")
    print(f"Val samples:   {len(df_val)}")
    print(f"Test samples:  {len(df_test)}")

    print("\nTrain Class Distribution (Study Level):")
    label_cols = [
        "Negative for Pneumonia",
        "Typical Appearance",
        "Indeterminate Appearance",
        "Atypical Appearance",
    ]
    # Since rows are images, we drop duplicates to count studies
    print(df_train.drop_duplicates("study_id")[label_cols].sum())

    print("\nVal Class Distribution (Study Level):")
    print(df_val.drop_duplicates("study_id")[label_cols].sum())

    # 2. Check File Existence
    for name, df in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
        check_files(name, df)

    # 3. Verify Split (Group Sampling)
    train_studies = set(df_train["study_id"].unique())
    val_studies = set(df_val["study_id"].unique())

    intersection = train_studies.intersection(val_studies)
    if intersection:
        raise AssertionError(
            f"Data Leakage Detected! {len(intersection)} studies found in both Train and Val sets."
        )
    else:
        print("\nSplit Verification Passed: No overlap between Train and Val studies.")


def check_files(dataset_name, df):
    # Randomly sample 1000 paths (or all if less than 1000)
    n_samples = min(1000, len(df))
    sample = df.sample(n=n_samples, random_state=RANDOM_STATE)

    missing_count = 0
    missing_samples = []

    for _, row in sample.iterrows():
        # Paths in metadata are relative to ./input
        full_path = os.path.join(INPUT_DIR, row["file_path"])
        if not os.path.exists(full_path):
            missing_count += 1
            if len(missing_samples) < 5:
                missing_samples.append(row["file_path"])

    ratio = missing_count / n_samples
    print(
        f"[{dataset_name}] Missing file ratio: {ratio:.4f} ({missing_count}/{n_samples})"
    )

    if ratio > 0.5:
        print("Sample missing paths:", missing_samples)
        raise FileNotFoundError(
            f"Too many files missing in {dataset_name} dataset! Ratio: {ratio}"
        )


if __name__ == "__main__":
    main()
