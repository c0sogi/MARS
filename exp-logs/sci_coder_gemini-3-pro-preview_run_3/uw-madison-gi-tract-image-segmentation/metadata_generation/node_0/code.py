import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

# Configuration
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
TRAIN_CSV = os.path.join(INPUT_DIR, "train.csv")
TEST_CSV = os.path.join(INPUT_DIR, "test.csv")
TRAIN_IMG_DIR = os.path.join(INPUT_DIR, "train")
TEST_IMG_DIR = os.path.join(INPUT_DIR, "test")
RANDOM_STATE = 42


def scan_images(root_path):
    """
    Traverses the directory structure to find image files and parse metadata from filenames.
    Returns a dictionary mapping ID -> image metadata.
    """
    image_map = {}
    if not os.path.exists(root_path):
        return image_map

    # Structure: caseXXX / caseXXX_dayYY / scans / slice_ZZZZ_w_h_pw_ph.png
    for case_dir in os.listdir(root_path):
        case_path = os.path.join(root_path, case_dir)
        if not os.path.isdir(case_path):
            continue

        for day_dir in os.listdir(case_path):
            day_path = os.path.join(case_path, day_dir)
            if not os.path.isdir(day_path):
                continue

            scans_path = os.path.join(day_path, "scans")
            if not os.path.isdir(scans_path):
                continue

            # Parse case and day strings for ID construction
            # case_dir ex: "case101"
            # day_dir ex: "case101_day20" -> we need "day20"
            c_part = case_dir
            d_part = day_dir.split("_")[-1]

            for f in os.listdir(scans_path):
                if f.endswith(".png"):
                    # Filename: slice_0001_266_266_1.50_1.50.png
                    # Format: slice_{id}_{w}_{h}_{pw}_{ph}.png
                    try:
                        parts = f.replace(".png", "").split("_")
                        if len(parts) >= 6:
                            slice_num = parts[1]
                            w = int(parts[2])
                            h = int(parts[3])
                            pw = float(parts[4])
                            ph = float(parts[5])

                            # Construct ID: case101_day20_slice_0001
                            constructed_id = f"{c_part}_{d_part}_slice_{slice_num}"

                            # Relative path from input dir
                            full_path = os.path.join(scans_path, f)
                            rel_path = os.path.relpath(full_path, INPUT_DIR)

                            image_map[constructed_id] = {
                                "id": constructed_id,
                                "case": c_part,
                                "day": d_part,
                                "slice": slice_num,
                                "image_path": rel_path,
                                "height": h,
                                "width": w,
                                "pixel_spacing_h": ph,
                                "pixel_spacing_w": pw,
                            }
                    except (ValueError, IndexError):
                        continue
    return image_map


def generate_metadata():
    os.makedirs(METADATA_DIR, exist_ok=True)

    print("Scanning image directories...")
    train_images = scan_images(TRAIN_IMG_DIR)
    test_images = scan_images(TEST_IMG_DIR)

    print(
        f"Found {len(train_images)} training images and {len(test_images)} test images."
    )

    # --- Process Training Data ---
    print("Processing train.csv...")
    df_train_raw = pd.read_csv(TRAIN_CSV)

    # Pivot to have one row per slice with columns for each class mask
    # train.csv cols: id, class, segmentation
    df_pivot = df_train_raw.pivot(
        index="id", columns="class", values="segmentation"
    ).reset_index()

    # Fill missing masks with empty strings (implies no mask)
    df_pivot = df_pivot.fillna("")

    # Convert image map to dataframe
    df_train_imgs = pd.DataFrame(list(train_images.values()))

    if df_train_imgs.empty:
        raise FileNotFoundError("No training images found in input directory.")

    # Merge segmentation data with image metadata
    # Inner join ensures we only include samples where we have both an image and an ID entry (or at least the image exists)
    # Note: If an image has no entry in train.csv, it might be dropped here.
    # However, usually train.csv covers the dataset. If we want to keep images without masks (as empty), we should use right join.
    # Given the task, we assume train.csv defines the training set.
    df_train_full = pd.merge(df_pivot, df_train_imgs, on="id", how="inner")

    # --- Split Train/Val ---
    print("Splitting training data...")
    splitter = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=RANDOM_STATE)

    # Split based on 'case' to avoid data leakage
    train_idx, val_idx = next(
        splitter.split(df_train_full, groups=df_train_full["case"])
    )

    train_set = df_train_full.iloc[train_idx].copy()
    val_set = df_train_full.iloc[val_idx].copy()

    # --- Process Test Data ---
    print("Processing test data...")
    # For test, we primarily rely on the images found on disk.
    df_test = pd.DataFrame(list(test_images.values()))

    # If test.csv exists, we can use it to filter, but usually for inference we just process available images.
    # We will just save the metadata for all found test images.

    # --- Save Metadata ---
    print("Saving metadata files...")
    train_set.to_csv(os.path.join(METADATA_DIR, "train.csv"), index=False)
    val_set.to_csv(os.path.join(METADATA_DIR, "val.csv"), index=False)
    df_test.to_csv(os.path.join(METADATA_DIR, "test.csv"), index=False)

    return train_set, val_set, df_test


def validate_datasets(train_df, val_df, test_df):
    print("\n=== Dataset Validation ===")

    # 1. Summary Statistics
    datasets = {"Train": train_df, "Validation": val_df, "Test": test_df}
    for name, df in datasets.items():
        print(f"Dataset: {name}")
        print(f"  Rows: {len(df)}")
        if not df.empty:
            if "case" in df.columns:
                print(f"  Unique Cases: {df['case'].nunique()}")
            if "large_bowel" in df.columns:
                non_empty = (df["large_bowel"] != "").sum()
                print(f"  Samples with Large Bowel mask: {non_empty}")
        print("-" * 30)

    # 2. Check File Paths
    def check_files(df, name):
        if df.empty:
            return

        # Sample 1000 paths
        sample_size = min(1000, len(df))
        sample = df.sample(n=sample_size, random_state=RANDOM_STATE)

        missing_count = 0
        missing_examples = []

        for _, row in sample.iterrows():
            rel_path = row["image_path"]
            # rel_path is relative to ./input
            full_path = os.path.join(INPUT_DIR, rel_path)

            if not os.path.exists(full_path):
                missing_count += 1
                if len(missing_examples) < 5:
                    missing_examples.append(rel_path)

        missing_ratio = missing_count / sample_size
        print(f"[{name}] Missing file ratio: {missing_ratio:.4f}")

        if missing_ratio > 0.5:
            print(f"Example missing files in {name}:")
            for p in missing_examples:
                print(f"  {p}")
            raise FileNotFoundError(
                f"Validation failed: Too many missing files in {name} dataset."
            )

    check_files(train_df, "Train")
    check_files(val_df, "Validation")
    check_files(test_df, "Test")

    # 3. Verify Split Integrity
    train_cases = set(train_df["case"].unique())
    val_cases = set(val_df["case"].unique())

    intersection = train_cases.intersection(val_cases)
    print(f"Split Intersection (Cases): {len(intersection)}")

    if len(intersection) > 0:
        raise AssertionError(
            f"Data Leakage Detected! Cases found in both Train and Val: {intersection}"
        )

    # Verify Ratio
    total_len = len(train_df) + len(val_df)
    if total_len > 0:
        val_ratio = len(val_df) / total_len
        print(f"Actual Validation Ratio: {val_ratio:.4f}")

    print("All validation checks passed.")


if __name__ == "__main__":
    try:
        train_df, val_df, test_df = generate_metadata()
        validate_datasets(train_df, val_df, test_df)
    except Exception as e:
        print(f"An error occurred: {e}")
        raise e
