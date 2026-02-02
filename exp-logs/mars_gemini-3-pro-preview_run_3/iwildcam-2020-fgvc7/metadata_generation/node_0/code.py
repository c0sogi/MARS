import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, GroupShuffleSplit

# --- Configuration ---
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
TRAIN_ANNOTATIONS_PATH = os.path.join(INPUT_DIR, "iwildcam2020_train_annotations.json")
TEST_INFO_PATH = os.path.join(INPUT_DIR, "iwildcam2020_test_information.json")
MEGADETECTOR_PATH = os.path.join(INPUT_DIR, "iwildcam2020_megadetector_results.json")
RANDOM_STATE = 42


def load_json(path):
    print(f"Loading {path}...")
    with open(path, "r") as f:
        return json.load(f)


def generate_metadata():
    if not os.path.exists(METADATA_DIR):
        os.makedirs(METADATA_DIR)

    # 1. Load Data
    train_data = load_json(TRAIN_ANNOTATIONS_PATH)
    test_data = load_json(TEST_INFO_PATH)
    md_data = load_json(MEGADETECTOR_PATH)

    # 2. Process MegaDetector Results
    # Extract ID and Max Confidence.
    # The MegaDetector file contains a list of images with detections.
    print("Processing MegaDetector results...")
    md_images = md_data.get("images", [])
    md_df = pd.DataFrame(md_images)

    # Keep only essential columns if they exist
    if "max_detection_conf" not in md_df.columns:
        md_df["max_detection_conf"] = 0.0

    # Ensure ID is string for merging
    md_df["id"] = md_df["id"].astype(str)
    md_df = md_df[["id", "max_detection_conf"]].rename(columns={"id": "image_id"})

    # 3. Process Training Data
    print("Processing Training Data...")
    train_images_df = pd.DataFrame(train_data["images"])
    train_annotations_df = pd.DataFrame(train_data["annotations"])

    # Ensure IDs are strings
    train_images_df["id"] = train_images_df["id"].astype(str)
    train_annotations_df["image_id"] = train_annotations_df["image_id"].astype(str)

    # Merge images with annotations
    # Use left join to keep images with no annotations (empty images)
    train_df = pd.merge(
        train_images_df,
        train_annotations_df,
        left_on="id",
        right_on="image_id",
        how="left",
    )

    # Handle empty images (NaN category_id implies no animal, which is category 0)
    train_df["category_id"] = train_df["category_id"].fillna(0).astype(int)

    # Construct file paths
    # If 'file_name' exists, use it. Otherwise assume id + .jpg
    if "file_name" in train_df.columns:
        train_df["file_path"] = train_df["file_name"].apply(
            lambda x: os.path.join("train", x)
        )
    else:
        train_df["file_path"] = train_df["id"].apply(
            lambda x: os.path.join("train", x + ".jpg")
        )

    # Merge with MegaDetector
    train_df = pd.merge(train_df, md_df, left_on="id", right_on="image_id", how="left")

    # Select relevant columns
    cols = ["id", "file_path", "category_id", "max_detection_conf"]
    # Add location or sequence info if available for splitting
    group_col = None
    if "location" in train_df.columns:
        cols.append("location")
        group_col = "location"
    elif "seq_id" in train_df.columns:
        cols.append("seq_id")
        group_col = "seq_id"

    train_df = train_df[cols].rename(columns={"id": "image_id"})

    # 4. Process Test Data
    print("Processing Test Data...")
    test_images_df = pd.DataFrame(test_data["images"])
    test_images_df["id"] = test_images_df["id"].astype(str)

    if "file_name" in test_images_df.columns:
        test_images_df["file_path"] = test_images_df["file_name"].apply(
            lambda x: os.path.join("test", x)
        )
    else:
        test_images_df["file_path"] = test_images_df["id"].apply(
            lambda x: os.path.join("test", x + ".jpg")
        )

    # Merge with MegaDetector
    test_df = pd.merge(
        test_images_df, md_df, left_on="id", right_on="image_id", how="left"
    )

    # Test metadata columns
    test_df = test_df[["id", "file_path", "max_detection_conf"]].rename(
        columns={"id": "image_id"}
    )

    # 5. Split Training Data
    print("Splitting Training Data...")
    if group_col:
        print(f"Using GroupShuffleSplit on '{group_col}'")
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=0.2, random_state=RANDOM_STATE
        )
        # We split based on the groups.
        train_idx, val_idx = next(
            splitter.split(
                train_df, train_df["category_id"], groups=train_df[group_col]
            )
        )
    else:
        print("Using StratifiedShuffleSplit on 'category_id'")
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=RANDOM_STATE
        )
        train_idx, val_idx = next(splitter.split(train_df, train_df["category_id"]))

    train_split = train_df.iloc[train_idx].copy()
    val_split = train_df.iloc[val_idx].copy()

    # 6. Save Metadata
    print("Saving metadata files...")
    train_split.to_csv(os.path.join(METADATA_DIR, "train_metadata.csv"), index=False)
    val_split.to_csv(os.path.join(METADATA_DIR, "val_metadata.csv"), index=False)
    test_df.to_csv(os.path.join(METADATA_DIR, "test_metadata.csv"), index=False)

    return group_col


def verify_metadata(group_col):
    print("\n--- Verifying Metadata ---")
    train_meta = pd.read_csv(os.path.join(METADATA_DIR, "train_metadata.csv"))
    val_meta = pd.read_csv(os.path.join(METADATA_DIR, "val_metadata.csv"))
    test_meta = pd.read_csv(os.path.join(METADATA_DIR, "test_metadata.csv"))

    # 1. Summary Statistics
    print(f"Train samples: {len(train_meta)}")
    print(f"Val samples: {len(val_meta)}")
    print(f"Test samples: {len(test_meta)}")
    print(f"Train shape: {train_meta.shape}")
    print(f"Val shape: {val_meta.shape}")

    print("\nClass Distribution (Train - Top 5):")
    print(train_meta["category_id"].value_counts().head())

    # 2. Verify Split Logic
    if group_col:
        train_groups = set(train_meta[group_col].unique())
        val_groups = set(val_meta[group_col].unique())
        overlap = train_groups.intersection(val_groups)
        print(f"\nUnique groups ({group_col}) in Train: {len(train_groups)}")
        print(f"Unique groups ({group_col}) in Val: {len(val_groups)}")

        if overlap:
            raise AssertionError(
                f"Group leakage detected! Overlapping groups: {overlap}"
            )
        print("Verification Passed: No group overlap between Train and Val.")

    # 3. Check File Paths
    def check_files(df, name):
        print(f"\nChecking file paths for {name} dataset...")
        # Sample 1000 paths
        sample_size = min(1000, len(df))
        sample_paths = (
            df["file_path"].sample(n=sample_size, random_state=RANDOM_STATE).tolist()
        )

        missing_count = 0
        missing_examples = []

        for rel_path in sample_paths:
            full_path = os.path.join(INPUT_DIR, rel_path)
            if not os.path.exists(full_path):
                missing_count += 1
                if len(missing_examples) < 5:
                    missing_examples.append(rel_path)

        ratio = missing_count / sample_size
        print(f"Missing file ratio: {ratio:.4f}")

        if ratio > 0.5:
            print(f"Examples of missing paths: {missing_examples}")
            raise FileNotFoundError(
                f"Missing file ratio ({ratio:.4f}) exceeds threshold (0.5) for {name} dataset."
            )

    check_files(train_meta, "Training")
    check_files(val_meta, "Validation")
    check_files(test_meta, "Test")

    print("\nAll verifications passed successfully.")


if __name__ == "__main__":
    group_column = generate_metadata()
    verify_metadata(group_column)
