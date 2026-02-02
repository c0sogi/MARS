import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
import random

# Constants
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
TRAIN_CSV_PATH = os.path.join(INPUT_DIR, "train.csv")
TEST_CSV_PATH = os.path.join(
    INPUT_DIR, "test.csv"
)  # Using test.csv as it defines the test set structure
RANDOM_STATE = 42


def parse_filename_info(filename):
    """
    Parses the filename to extract slice number, width, height, and spacing.
    Format: slice_{number}_{width}_{height}_{spacing_w}_{spacing_h}.png
    """
    # Remove extension
    base = os.path.splitext(filename)[0]
    parts = base.split("_")

    # Expected parts: ['slice', '0001', '360', '310', '1.50', '1.50']
    if len(parts) < 6:
        return None

    slice_num = parts[1]
    width = int(parts[2])
    height = int(parts[3])
    spacing_w = float(parts[4])
    spacing_h = float(parts[5])

    return {
        "slice_id": slice_num,
        "img_width": width,
        "img_height": height,
        "pixel_spacing_w": spacing_w,
        "pixel_spacing_h": spacing_h,
    }


def scan_directory(subfolder):
    """
    Scans input/{subfolder} (e.g., 'train' or 'test') to build a dataframe of file paths and metadata.
    Constructs the 'id' column to match the CSV format: case{case}_day{day}_slice_{slice}
    """
    root_path = os.path.join(INPUT_DIR, subfolder)
    data = []

    # Walk through the directory
    # Structure: caseXXX / caseXXX_dayYY / scans / slice_....png
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.endswith(".png"):
                # Extract path components
                # root ends with .../caseXXX_dayYY/scans
                parent_dir = os.path.basename(os.path.dirname(root))  # caseXXX_dayYY

                # Parse case and day from directory name
                # Format: case123_day20
                try:
                    case_str, day_str = parent_dir.split("_")
                    case_id = case_str.replace("case", "")
                    day_id = day_str.replace("day", "")
                except ValueError:
                    continue  # Skip if folder structure doesn't match

                # Parse filename info
                info = parse_filename_info(file)
                if info is None:
                    continue

                # Construct ID: case{case}_day{day}_slice_{slice}
                # Note: The CSV IDs use the slice number from the filename (e.g., 0001)
                img_id = f"{case_str}_{day_str}_slice_{info['slice_id']}"

                # Relative path from ./input
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, INPUT_DIR)

                row = {
                    "id": img_id,
                    "file_path": rel_path,
                    "case": int(case_id),
                    "day": int(day_id),
                    "slice": int(info["slice_id"]),
                    "img_width": info["img_width"],
                    "img_height": info["img_height"],
                    "pixel_spacing_w": info["pixel_spacing_w"],
                    "pixel_spacing_h": info["pixel_spacing_h"],
                }
                data.append(row)

    return pd.DataFrame(data)


def verify_files(df, name):
    """
    Verifies that a sample of file paths in the dataframe exist.
    """
    if "file_path" not in df.columns:
        return

    paths = df["file_path"].unique()
    n_check = min(1000, len(paths))
    if n_check == 0:
        return

    # Randomly select paths
    sample_paths = np.random.choice(paths, n_check, replace=False)

    missing_count = 0
    missing_samples = []

    for p in sample_paths:
        full_path = os.path.join(INPUT_DIR, p)
        if not os.path.exists(full_path):
            missing_count += 1
            if len(missing_samples) < 5:
                missing_samples.append(p)

    ratio = missing_count / n_check
    print(f"[{name}] Missing file ratio: {ratio:.4f} ({missing_count}/{n_check})")

    if len(missing_samples) > 0:
        print(f"[{name}] Sample missing files: {missing_samples}")

    if ratio > 0.5:
        raise FileNotFoundError(f"Too many missing files in {name} metadata.")


def main():
    print("Starting metadata generation...")

    # 1. Scan Directories
    print("Scanning train directory...")
    train_files_df = scan_directory("train")
    print(f"Found {len(train_files_df)} images in train.")

    print("Scanning test directory...")
    test_files_df = scan_directory("test")
    print(f"Found {len(test_files_df)} images in test.")

    # 2. Load CSVs
    print("Loading CSV files...")
    train_csv = pd.read_csv(TRAIN_CSV_PATH)
    test_csv = pd.read_csv(TEST_CSV_PATH)

    # 3. Merge Metadata
    # train.csv is long format (multiple rows per image id for different classes)
    # We merge file info into the CSV data.
    print("Merging train metadata...")
    train_full = pd.merge(train_csv, train_files_df, on="id", how="inner")

    # Check if we lost data during merge
    if len(train_full) == 0:
        raise ValueError("Merge resulted in empty train dataset. Check ID formats.")

    # test.csv also needs file info
    print("Merging test metadata...")
    test_full = pd.merge(test_csv, test_files_df, on="id", how="inner")

    # 4. Split Train/Val
    # We must split by 'case' to avoid data leakage (Group Sampling)
    print("Splitting train/validation...")
    splitter = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=RANDOM_STATE)

    # We split based on unique cases
    groups = train_full["case"]
    train_idx, val_idx = next(splitter.split(train_full, groups=groups))

    train_meta = train_full.iloc[train_idx].copy()
    val_meta = train_full.iloc[val_idx].copy()

    # 5. Save Metadata
    os.makedirs(METADATA_DIR, exist_ok=True)

    train_meta_path = os.path.join(METADATA_DIR, "train_metadata.csv")
    val_meta_path = os.path.join(METADATA_DIR, "val_metadata.csv")
    test_meta_path = os.path.join(METADATA_DIR, "test_metadata.csv")

    train_meta.to_csv(train_meta_path, index=False)
    val_meta.to_csv(val_meta_path, index=False)
    test_full.to_csv(test_meta_path, index=False)

    print(f"Saved metadata to {METADATA_DIR}")

    # 6. Verification and Statistics
    print("\n=== Verification & Statistics ===")

    # Load back data to verify
    df_train = pd.read_csv(train_meta_path)
    df_val = pd.read_csv(val_meta_path)
    df_test = pd.read_csv(test_meta_path)

    # Stats
    print(
        f"Train set: {len(df_train)} rows, {df_train['id'].nunique()} unique images, {df_train['case'].nunique()} unique cases."
    )
    print(
        f"Val set:   {len(df_val)} rows, {df_val['id'].nunique()} unique images, {df_val['case'].nunique()} unique cases."
    )
    print(f"Test set:  {len(df_test)} rows, {df_test['id'].nunique()} unique images.")

    # Class distribution
    print("\nTrain Class Distribution:")
    print(df_train["class"].value_counts())
    print("\nVal Class Distribution:")
    print(df_val["class"].value_counts())

    # Verify Split Logic (Disjoint Cases)
    train_cases = set(df_train["case"].unique())
    val_cases = set(df_val["case"].unique())
    intersection = train_cases.intersection(val_cases)

    if len(intersection) > 0:
        raise AssertionError(
            f"Data Leakage detected! Cases {intersection} are in both train and val."
        )
    print("\nSplit verification passed: No case overlap between train and val.")

    # Verify File Existence
    verify_files(df_train, "Train")
    verify_files(df_val, "Val")
    verify_files(df_test, "Test")

    print("\nAll checks passed successfully.")


if __name__ == "__main__":
    main()
