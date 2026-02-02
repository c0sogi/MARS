import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import random

# Constants
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
RANDOM_STATE = 42
TRAIN_ANNOTATIONS_FILE = "iwildcam2020_train_annotations.json"
TEST_INFO_FILE = "iwildcam2020_test_information.json"


def load_json(filepath):
    with open(filepath, "r") as f:
        return json.load(f)


def generate_metadata():
    print("Starting metadata generation...")

    # Ensure metadata directory exists
    os.makedirs(METADATA_DIR, exist_ok=True)

    # 1. Load Training Data
    train_json_path = os.path.join(INPUT_DIR, TRAIN_ANNOTATIONS_FILE)
    if not os.path.exists(train_json_path):
        raise FileNotFoundError(f"Training annotations not found at {train_json_path}")

    print(f"Loading {train_json_path}...")
    train_data = load_json(train_json_path)

    # Map image_id to category_id
    # Note: An image might have multiple annotations. We take the first one found.
    # Images with no annotations are assumed to be category 0 (empty) if not present.
    img_to_category = {}
    for ann in train_data.get("annotations", []):
        img_id = ann["image_id"]
        if img_id not in img_to_category:
            img_to_category[img_id] = ann["category_id"]

    # Create Train DataFrame
    train_rows = []
    for img in train_data.get("images", []):
        img_id = img["id"]
        file_name = img.get("file_name", f"{img_id}.jpg")

        # Ensure path is relative to input/ and points to train/ directory
        # The raw data structure usually has images in input/train/
        # If file_name already contains 'train/', use it, else prepend
        if not file_name.startswith("train/"):
            file_path = os.path.join("train", file_name)
        else:
            file_path = file_name

        # Get category, default to 0 if not annotated
        category_id = img_to_category.get(img_id, 0)

        train_rows.append(
            {"image_id": img_id, "file_path": file_path, "category_id": category_id}
        )

    full_train_df = pd.DataFrame(train_rows)
    print(f"Total training samples loaded: {len(full_train_df)}")

    # 2. Load Test Data
    test_json_path = os.path.join(INPUT_DIR, TEST_INFO_FILE)
    if not os.path.exists(test_json_path):
        raise FileNotFoundError(f"Test info not found at {test_json_path}")

    print(f"Loading {test_json_path}...")
    test_data = load_json(test_json_path)

    test_rows = []
    for img in test_data.get("images", []):
        img_id = img["id"]
        file_name = img.get("file_name", f"{img_id}.jpg")

        if not file_name.startswith("test/"):
            file_path = os.path.join("test", file_name)
        else:
            file_path = file_name

        test_rows.append({"image_id": img_id, "file_path": file_path})

    test_df = pd.DataFrame(test_rows)
    print(f"Total test samples loaded: {len(test_df)}")

    # 3. Stratified Split (Train/Val)
    # Handle rare classes (count < 2) which cannot be stratified
    class_counts = full_train_df["category_id"].value_counts()
    rare_classes = class_counts[class_counts < 2].index.tolist()

    print(f"Found {len(rare_classes)} rare classes with < 2 samples.")

    rare_df = full_train_df[full_train_df["category_id"].isin(rare_classes)]
    common_df = full_train_df[~full_train_df["category_id"].isin(rare_classes)]

    print(f"Splitting {len(common_df)} common samples into Train/Val (80:20)...")

    # Stratified split on common data
    train_common, val_common = train_test_split(
        common_df,
        test_size=0.2,
        stratify=common_df["category_id"],
        random_state=RANDOM_STATE,
    )

    # Combine rare data back into training set
    train_df = (
        pd.concat([train_common, rare_df])
        .sample(frac=1, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )
    val_df = val_common.reset_index(drop=True)

    # 4. Save Metadata
    print("Saving metadata files...")
    train_csv_path = os.path.join(METADATA_DIR, "train.csv")
    val_csv_path = os.path.join(METADATA_DIR, "val.csv")
    test_csv_path = os.path.join(METADATA_DIR, "test.csv")

    train_df.to_csv(train_csv_path, index=False)
    val_df.to_csv(val_csv_path, index=False)
    test_df.to_csv(test_csv_path, index=False)

    return train_df, val_df, test_df


def verify_metadata(train_df, val_df, test_df):
    print("\n--- Verifying Metadata ---")

    # 1. Summary Statistics
    print(f"Train Set: {len(train_df)} samples")
    print(f"Val Set:   {len(val_df)} samples")
    print(f"Test Set:  {len(test_df)} samples")

    n_classes_train = train_df["category_id"].nunique()
    n_classes_val = val_df["category_id"].nunique()
    print(f"Unique classes in Train: {n_classes_train}")
    print(f"Unique classes in Val:   {n_classes_val}")

    # 2. File Path Verification
    print("Checking file paths...")

    def check_paths(df, name):
        if len(df) == 0:
            return
        sample_size = min(1000, len(df))
        samples = df.sample(n=sample_size, random_state=RANDOM_STATE)[
            "file_path"
        ].tolist()

        missing_count = 0
        missing_samples = []
        for p in samples:
            full_path = os.path.join(INPUT_DIR, p)
            if not os.path.exists(full_path):
                missing_count += 1
                if len(missing_samples) < 5:
                    missing_samples.append(p)

        missing_ratio = missing_count / sample_size
        print(
            f"[{name}] Missing file ratio: {missing_ratio:.4f} ({missing_count}/{sample_size})"
        )

        if missing_ratio > 0.5:
            print("Sample missing files:", missing_samples)
            raise FileNotFoundError(
                f"Too many missing files in {name} dataset! Ratio: {missing_ratio}"
            )

    check_paths(train_df, "Train")
    check_paths(val_df, "Val")
    check_paths(test_df, "Test")

    # 3. Validation Split Verification
    print("Verifying split requirements...")

    # Check if validation set exists
    if len(val_df) == 0:
        raise AssertionError("Validation set is empty!")

    # Check stratification roughly
    # We expect the distribution of classes in Val to be similar to Train (for common classes)
    # Since we handled rare classes separately, we check that val contains a subset of train classes
    train_classes = set(train_df["category_id"].unique())
    val_classes = set(val_df["category_id"].unique())

    if not val_classes.issubset(train_classes):
        diff = val_classes - train_classes
        raise AssertionError(
            f"Validation set contains classes not in training set: {diff}"
        )

    print("Verification passed successfully.")


if __name__ == "__main__":
    try:
        train_df, val_df, test_df = generate_metadata()
        verify_metadata(train_df, val_df, test_df)
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        raise
