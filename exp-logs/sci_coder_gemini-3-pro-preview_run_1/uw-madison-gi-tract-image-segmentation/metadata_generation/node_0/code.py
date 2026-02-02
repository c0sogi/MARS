import os
import glob
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

# Constants
INPUT_DIR = "./input"
TRAIN_DIR = os.path.join(INPUT_DIR, "train")
TEST_DIR = os.path.join(INPUT_DIR, "test")
METADATA_DIR = "./metadata"
RANDOM_STATE = 42


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def parse_directory(root_dir, subset_name):
    """
    Walks through the directory structure to find image files and extract metadata.
    Expected structure: caseXXX/caseXXX_dayYY/scans/slice_ZZZZ_w_h_px_py.png
    """
    data = []
    # Pattern to match the filename and extract metadata
    # filename example: slice_0001_360_310_1.50_1.50.png
    # pattern: slice_{slice_id}_{width}_{height}_{spacing_x}_{spacing_y}.png
    file_pattern = re.compile(r"slice_(\d+)_(\d+)_(\d+)_([\d\.]+)_([\d\.]+)\.png")

    # We walk the directory
    print(f"Scanning {subset_name} directory: {root_dir}...")
    files_found = 0

    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".png"):
                match = file_pattern.match(file)
                if match:
                    slice_id_str, w, h, sp_x, sp_y = match.groups()

                    # Extract case and day from the directory path
                    # Path ends in .../caseXXX/caseXXX_dayYY/scans
                    path_parts = os.path.normpath(root).split(os.sep)

                    # We expect structure like: .../case110/case110_day12/scans
                    # So parent is case110_day12
                    if len(path_parts) >= 2:
                        case_day_str = path_parts[-2]  # e.g., case110_day12

                        # Parse case and day
                        # case_day_str format: case{id}_day{num}
                        cd_parts = case_day_str.split("_")
                        if len(cd_parts) >= 2:
                            case_str = cd_parts[0]  # case110
                            day_str = cd_parts[1]  # day12

                            # Construct the ID used in train.csv: caseXXX_dayYY_slice_ZZZZ
                            # Note: slice_id in csv usually has 4 digits, matched by regex \d+
                            # We keep the slice_id string as found in filename (usually 0001)
                            # but ensure format matches standard if needed.
                            # Looking at sample: case123_day20_slice_0001

                            generated_id = f"{case_str}_{day_str}_slice_{slice_id_str}"

                            rel_path = os.path.relpath(
                                os.path.join(root, file), INPUT_DIR
                            )

                            data.append(
                                {
                                    "id": generated_id,
                                    "case": case_str,
                                    "day": day_str,
                                    "slice": slice_id_str,
                                    "width": int(w),
                                    "height": int(h),
                                    "spacing_x": float(sp_x),
                                    "spacing_y": float(sp_y),
                                    "file_path": rel_path,
                                }
                            )
                            files_found += 1

    print(f"Found {files_found} images in {subset_name}.")
    return pd.DataFrame(data)


def process_annotations(train_csv_path):
    """
    Loads train.csv, pivots it to wide format (one row per slice).
    """
    print("Loading and processing annotations...")
    df = pd.read_csv(train_csv_path)

    # Check columns
    required_cols = {"id", "class", "segmentation"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"train.csv missing columns. Found: {df.columns}")

    # Pivot: id as index, class as columns, segmentation as values
    # Fill NaN with empty string (no mask)
    df_pivoted = df.pivot(index="id", columns="class", values="segmentation").fillna("")

    # Reset index to make 'id' a column again
    df_pivoted.reset_index(inplace=True)

    # Ensure all expected classes are present
    expected_classes = ["large_bowel", "small_bowel", "stomach"]
    for c in expected_classes:
        if c not in df_pivoted.columns:
            df_pivoted[c] = ""

    return df_pivoted


def main():
    ensure_dir(METADATA_DIR)

    # 1. Scan Directories
    df_train_imgs = parse_directory(TRAIN_DIR, "train")
    df_test_imgs = parse_directory(TEST_DIR, "test")

    # 2. Process Annotations
    train_csv_path = os.path.join(INPUT_DIR, "train.csv")
    if os.path.exists(train_csv_path):
        df_annotations = process_annotations(train_csv_path)

        # Merge image info with annotations
        # We use a left join on images to ensure we only keep entries with valid files
        # If an image has no entry in train.csv, it implies no masks (all empty)
        df_train_merged = pd.merge(df_train_imgs, df_annotations, on="id", how="left")

        # Fill NaNs for mask columns (in case image existed but not in train.csv)
        for c in ["large_bowel", "small_bowel", "stomach"]:
            if c in df_train_merged.columns:
                df_train_merged[c] = df_train_merged[c].fillna("")
    else:
        print(
            "Warning: train.csv not found. Creating dummy train metadata based on images only."
        )
        df_train_merged = df_train_imgs
        for c in ["large_bowel", "small_bowel", "stomach"]:
            df_train_merged[c] = ""

    # 3. Split Train/Validation
    # We must split by 'case' to avoid leakage
    print("Splitting data into train and validation sets...")
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_STATE)

    # We need groups based on 'case'
    groups = df_train_merged["case"]

    train_idx, val_idx = next(splitter.split(df_train_merged, groups=groups))

    df_train = df_train_merged.iloc[train_idx].copy()
    df_val = df_train_merged.iloc[val_idx].copy()

    # 4. Save Metadata
    print("Saving metadata...")
    train_save_path = os.path.join(METADATA_DIR, "train.csv")
    val_save_path = os.path.join(METADATA_DIR, "val.csv")
    test_save_path = os.path.join(METADATA_DIR, "test.csv")

    df_train.to_csv(train_save_path, index=False)
    df_val.to_csv(val_save_path, index=False)

    # For test, we just save the image info.
    # If sample_submission is needed to filter, we could use it,
    # but usually providing all available test images is safer for inference.
    df_test_imgs.to_csv(test_save_path, index=False)

    print("Metadata generation complete.")

    # 5. Validation and Checks
    validate_outputs(df_train, df_val, df_test_imgs)


def validate_outputs(df_train, df_val, df_test):
    print("\n=== Validation & Summary ===")

    # Summary Statistics
    print(f"Train set shape: {df_train.shape}")
    print(f"Validation set shape: {df_val.shape}")
    print(f"Test set shape: {df_test.shape}")

    print(f"Train unique cases: {df_train['case'].nunique()}")
    print(f"Val unique cases: {df_val['case'].nunique()}")

    # Check Class Distribution (count of non-empty masks)
    if "large_bowel" in df_train.columns:
        for col in ["large_bowel", "small_bowel", "stomach"]:
            train_pos = (df_train[col] != "").sum()
            val_pos = (df_val[col] != "").sum()
            print(
                f"Class '{col}': Train={train_pos} ({train_pos/len(df_train):.2%}), Val={val_pos} ({val_pos/len(df_val):.2%})"
            )

    # Check 1: File Path Existence
    print("\nChecking file path existence (sampling 1000 files)...")

    # Combine all paths to sample from
    all_paths = pd.concat(
        [df_train["file_path"], df_val["file_path"], df_test["file_path"]]
    )

    if len(all_paths) > 0:
        sample_size = min(1000, len(all_paths))
        sample_paths = all_paths.sample(n=sample_size, random_state=RANDOM_STATE)

        missing_count = 0
        missing_samples = []

        for rel_path in sample_paths:
            full_path = os.path.join(INPUT_DIR, rel_path)
            if not os.path.exists(full_path):
                missing_count += 1
                if len(missing_samples) < 5:
                    missing_samples.append(rel_path)

        missing_ratio = missing_count / sample_size
        print(
            f"Missing file ratio: {missing_ratio:.4f} ({missing_count}/{sample_size})"
        )

        if missing_count > 0:
            print("Sample missing files:")
            for p in missing_samples:
                print(f"  {p}")

        if missing_ratio > 0.5:
            raise FileNotFoundError(
                f"High missing file ratio detected: {missing_ratio}"
            )
    else:
        print("No files found to check.")

    # Check 2: Validation Split Integrity (Group Leakage)
    print("\nChecking validation split integrity...")
    train_cases = set(df_train["case"].unique())
    val_cases = set(df_val["case"].unique())

    intersection = train_cases.intersection(val_cases)
    if intersection:
        raise AssertionError(
            f"Data leakage detected! Cases in both train and val: {intersection}"
        )
    else:
        print("Success: No case overlap between train and validation sets.")

    # Check 3: Stratification/Group Split Logic
    # We used GroupShuffleSplit, so we expect roughly 80/20 split on groups (cases), not necessarily on rows
    # because cases have different numbers of slices.
    n_train_cases = len(train_cases)
    n_val_cases = len(val_cases)
    total_cases = n_train_cases + n_val_cases

    if total_cases > 0:
        val_case_ratio = n_val_cases / total_cases
        print(
            f"Case split ratio: Train={n_train_cases}, Val={n_val_cases} (Val Ratio={val_case_ratio:.2f})"
        )

        # Allow some tolerance because number of cases might be small
        if not (0.1 < val_case_ratio < 0.3):
            print(
                "Warning: Validation case ratio deviates significantly from 0.2. This might be due to small dataset size."
            )

    print("\nAll checks passed successfully.")


if __name__ == "__main__":
    main()
