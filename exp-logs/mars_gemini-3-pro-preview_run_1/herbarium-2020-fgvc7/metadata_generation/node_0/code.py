import json
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path

# Constants
INPUT_DIR = Path("./input")
METADATA_DIR = Path("./metadata")
TRAIN_META_PATH = INPUT_DIR / "nybg2020/train/metadata.json"
TEST_META_PATH = INPUT_DIR / "nybg2020/test/metadata.json"
RANDOM_STATE = 42


def load_json(path):
    print(f"Loading {path}...")
    with open(path, "r") as f:
        return json.load(f)


def process_train_metadata():
    data = load_json(TRAIN_META_PATH)

    # Create DataFrames from lists of dictionaries
    print("Parsing training images...")
    images_df = pd.DataFrame(data["images"])
    print("Parsing training annotations...")
    annotations_df = pd.DataFrame(data["annotations"])

    # Rename id columns to avoid confusion during merge
    # images: id -> image_id
    # annotations: image_id is already present
    images_df = images_df.rename(columns={"id": "image_id"})

    # Merge images and annotations
    print("Merging training data...")
    merged_df = pd.merge(annotations_df, images_df, on="image_id", how="left")

    # Construct relative file path
    # The file_name in json is like "images/000/00/495523.jpg"
    # The physical path is input/nybg2020/train/images/000/00/495523.jpg
    # We want path relative to ./input, so: "nybg2020/train/" + file_name
    merged_df["file_path"] = "nybg2020/train/" + merged_df["file_name"]

    # Select relevant columns
    # We keep category_id as the target label
    cols = ["image_id", "file_path", "category_id", "region_id"]
    return merged_df[cols]


def process_test_metadata():
    data = load_json(TEST_META_PATH)

    print("Parsing test images...")
    images_df = pd.DataFrame(data["images"])

    # Rename id -> image_id for consistency
    images_df = images_df.rename(columns={"id": "image_id"})

    # Construct relative file path
    # Physical path: input/nybg2020/test/images/...
    images_df["file_path"] = "nybg2020/test/" + images_df["file_name"]

    cols = ["image_id", "file_path"]
    return images_df[cols]


def split_data(df):
    print("Splitting data into train and validation sets...")

    # Handle classes with too few samples for stratification
    class_counts = df["category_id"].value_counts()
    rare_classes = class_counts[class_counts < 2].index

    rare_mask = df["category_id"].isin(rare_classes)
    rare_df = df[rare_mask]
    rest_df = df[~rare_mask]

    print(
        f"Found {len(rare_df)} samples belonging to {len(rare_classes)} rare classes (count < 2). Adding to train set."
    )

    # Stratified split on the rest
    train_rest, val_rest = train_test_split(
        rest_df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=rest_df["category_id"],
    )

    # Combine rare samples back into train
    train_final = (
        pd.concat([train_rest, rare_df], axis=0)
        .sample(frac=1, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )
    val_final = val_rest.reset_index(drop=True)

    return train_final, val_final


def check_file_existence(df, name):
    print(f"Checking file existence for {name}...")
    sample_size = min(1000, len(df))
    sample = df.sample(n=sample_size, random_state=RANDOM_STATE)

    missing_count = 0
    missing_samples = []

    for _, row in sample.iterrows():
        full_path = INPUT_DIR / row["file_path"]
        if not full_path.exists():
            missing_count += 1
            if len(missing_samples) < 5:
                missing_samples.append(str(full_path))

    missing_ratio = missing_count / sample_size
    print(
        f"{name}: Missing file ratio = {missing_ratio:.4f} ({missing_count}/{sample_size})"
    )

    if missing_ratio > 0.5:
        print("Sample missing files:", missing_samples)
        raise FileNotFoundError(
            f"More than 50% of files are missing in {name} dataset."
        )


def verify_stratification(train_df, val_df):
    print("Verifying stratification...")
    # Calculate distribution
    train_dist = train_df["category_id"].value_counts(normalize=True)
    val_dist = val_df["category_id"].value_counts(normalize=True)

    # Align indices
    all_cats = set(train_dist.index) | set(val_dist.index)

    # We only check for categories present in validation (since we forced singletons into train)
    # The distribution should be roughly similar for common classes.
    # We'll check the correlation or mean absolute difference for classes present in both.

    common_cats = train_dist.index.intersection(val_dist.index)

    diffs = (train_dist[common_cats] - val_dist[common_cats]).abs()
    mean_diff = diffs.mean()

    print(
        f"Mean absolute difference in class probabilities for common classes: {mean_diff:.6f}"
    )

    # A loose check, as rare classes distort the distribution slightly
    if mean_diff > 0.01:
        raise AssertionError(
            f"Stratification failed. Mean difference {mean_diff} is too high."
        )
    print("Stratification verification passed.")


def main():
    # Ensure metadata directory exists
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Process Data
    full_train_df = process_train_metadata()
    test_df = process_test_metadata()

    # 2. Split Data
    train_df, val_df = split_data(full_train_df)

    # 3. Save Metadata
    print("Saving metadata files...")
    train_df.to_csv(METADATA_DIR / "train.csv", index=False)
    val_df.to_csv(METADATA_DIR / "val.csv", index=False)
    test_df.to_csv(METADATA_DIR / "test.csv", index=False)

    # 4. Verification
    print("\n--- Verification ---")

    # Load back to ensure integrity
    train_loaded = pd.read_csv(METADATA_DIR / "train.csv")
    val_loaded = pd.read_csv(METADATA_DIR / "val.csv")
    test_loaded = pd.read_csv(METADATA_DIR / "test.csv")

    # Print Stats
    print(f"Train samples: {len(train_loaded)}")
    print(f"Val samples: {len(val_loaded)}")
    print(f"Test samples: {len(test_loaded)}")
    print(f"Unique classes in Train: {train_loaded['category_id'].nunique()}")
    print(f"Unique classes in Val: {val_loaded['category_id'].nunique()}")

    # Check file existence
    check_file_existence(train_loaded, "Train")
    check_file_existence(val_loaded, "Val")
    check_file_existence(test_loaded, "Test")

    # Verify Stratification
    verify_stratification(train_loaded, val_loaded)

    print("\nMetadata generation and verification completed successfully.")


if __name__ == "__main__":
    main()
