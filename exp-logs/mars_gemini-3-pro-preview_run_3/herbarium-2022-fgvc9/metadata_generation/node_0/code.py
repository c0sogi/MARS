import os
import json
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def main():
    # Define directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    os.makedirs(METADATA_DIR, exist_ok=True)

    # Constants
    RANDOM_STATE = 42
    VAL_SIZE = 0.2

    print("Starting metadata generation...")

    # ---------------------------------------------------------
    # 1. Process Training Data
    # ---------------------------------------------------------
    print("Scanning training images...")
    # The description and file listing indicate a structured folder system.
    # We scan the directory to ensure we get all actual existing files.
    # Pattern: input/train_images/*/*/*.jpg
    train_image_pattern = os.path.join(INPUT_DIR, "train_images", "*", "*", "*.jpg")
    train_files = glob.glob(train_image_pattern)

    train_data = []
    for filepath in train_files:
        # Get path relative to input directory
        rel_path = os.path.relpath(filepath, INPUT_DIR)
        filename = os.path.basename(filepath)

        # Extract category_id from filename
        # Format: {category_id}__{image_num}.jpg (e.g., 00000__001.jpg)
        try:
            category_part = filename.split("__")[0]
            category_id = int(category_part)

            # Use filename without extension as image_id for training data
            image_id = os.path.splitext(filename)[0]

            train_data.append(
                {
                    "image_id": image_id,
                    "category_id": category_id,
                    "file_path": rel_path,
                }
            )
        except (ValueError, IndexError):
            # Skip files that don't match the expected format
            continue

    df_full_train = pd.DataFrame(train_data)
    print(f"Found {len(df_full_train)} training images.")

    # ---------------------------------------------------------
    # 2. Process Test Data
    # ---------------------------------------------------------
    print("Loading test metadata...")
    test_meta_path = os.path.join(INPUT_DIR, "test_metadata.json")

    with open(test_meta_path, "r") as f:
        test_meta_json = json.load(f)

    test_data = []
    for item in test_meta_json:
        # Item has 'file_name', 'image_id', 'license'
        # file_name in json is like "000/test-000000.jpg"
        # Actual path is test_images/000/test-000000.jpg
        rel_path = os.path.join("test_images", item["file_name"])
        test_data.append({"image_id": item["image_id"], "file_path": rel_path})

    df_test = pd.DataFrame(test_data)
    print(f"Found {len(df_test)} test images.")

    # ---------------------------------------------------------
    # 3. Create Validation Split
    # ---------------------------------------------------------
    print("Creating stratified validation split...")

    if df_full_train.empty:
        raise ValueError("No training data found. Check input directory structure.")

    # Identify classes with too few samples for stratification
    class_counts = df_full_train["category_id"].value_counts()
    rare_classes = class_counts[class_counts < 2].index

    # Split data into rare (force train) and common (stratify)
    df_rare = df_full_train[df_full_train["category_id"].isin(rare_classes)].copy()
    df_common = df_full_train[~df_full_train["category_id"].isin(rare_classes)].copy()

    print(f"Classes with < 2 samples: {len(rare_classes)}")

    # Perform stratified split on common classes
    if len(df_common) > 0:
        train_common, val_common = train_test_split(
            df_common,
            test_size=VAL_SIZE,
            stratify=df_common["category_id"],
            random_state=RANDOM_STATE,
        )

        # Combine rare classes back into training set
        df_train = (
            pd.concat([train_common, df_rare], axis=0)
            .sample(frac=1, random_state=RANDOM_STATE)
            .reset_index(drop=True)
        )
        df_val = val_common.reset_index(drop=True)
    else:
        # If all classes are rare (unlikely), put everything in train
        df_train = df_rare
        df_val = pd.DataFrame(columns=df_rare.columns)
        print("Warning: No classes suitable for stratified validation split.")

    # ---------------------------------------------------------
    # 4. Save Metadata
    # ---------------------------------------------------------
    print("Saving metadata files...")
    df_train.to_csv(os.path.join(METADATA_DIR, "train.csv"), index=False)
    df_val.to_csv(os.path.join(METADATA_DIR, "val.csv"), index=False)
    df_test.to_csv(os.path.join(METADATA_DIR, "test.csv"), index=False)

    # ---------------------------------------------------------
    # 5. Validation and Checks
    # ---------------------------------------------------------
    print("\n==== Dataset Statistics ====")
    print(
        f"Train Set: {len(df_train)} samples, {df_train['category_id'].nunique()} classes"
    )
    print(
        f"Val Set:   {len(df_val)} samples, {df_val['category_id'].nunique()} classes"
    )
    print(f"Test Set:  {len(df_test)} samples")

    def check_file_existence(df, name):
        if len(df) == 0:
            return

        # Check up to 1000 random files
        sample_size = min(1000, len(df))
        sample_paths = (
            df["file_path"].sample(n=sample_size, random_state=RANDOM_STATE).tolist()
        )

        missing_count = 0
        missing_examples = []

        for p in sample_paths:
            full_path = os.path.join(INPUT_DIR, p)
            if not os.path.exists(full_path):
                missing_count += 1
                if len(missing_examples) < 5:
                    missing_examples.append(p)

        missing_ratio = missing_count / sample_size
        print(
            f"[{name}] Missing file ratio: {missing_ratio:.4f} ({missing_count}/{sample_size})"
        )

        if missing_ratio > 0.5:
            print(f"Example missing files in {name}:")
            for mp in missing_examples:
                print(f" - {mp}")
            raise FileNotFoundError(
                f"Critical error: More than 50% of files missing in {name} dataset."
            )

    print("\nVerifying file paths...")
    check_file_existence(df_train, "Train")
    check_file_existence(df_val, "Validation")
    check_file_existence(df_test, "Test")

    # Verify split integrity
    if len(df_val) > 0:
        # Check for data leakage
        train_ids = set(df_train["image_id"])
        val_ids = set(df_val["image_id"])
        intersection = train_ids.intersection(val_ids)
        assert (
            len(intersection) == 0
        ), f"Data leakage detected: {len(intersection)} IDs in both train and val."
        print(
            "Split integrity check passed: No overlap between train and validation sets."
        )

    print("\nMetadata generation completed successfully.")


if __name__ == "__main__":
    main()
