import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import random


def main():
    # Configuration
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    TRAIN_META_PATH = os.path.join(INPUT_DIR, "nybg2020/train/metadata.json")
    TEST_META_PATH = os.path.join(INPUT_DIR, "nybg2020/test/metadata.json")

    os.makedirs(METADATA_DIR, exist_ok=True)

    print("Loading training metadata...")
    with open(TRAIN_META_PATH, "r") as f:
        train_data = json.load(f)

    print("Loading test metadata...")
    with open(TEST_META_PATH, "r") as f:
        test_data = json.load(f)

    # --- Process Training Data ---
    print("Processing training data...")
    images_df = pd.DataFrame(train_data["images"])
    annotations_df = pd.DataFrame(train_data["annotations"])

    # Ensure IDs are integers for merging
    images_df["id"] = images_df["id"].astype(int)
    annotations_df["image_id"] = annotations_df["image_id"].astype(int)

    # Merge images and annotations
    # annotations contain 'category_id', 'id' (annotation id), 'image_id', 'region_id'
    # images contain 'file_name', 'height', 'id' (image id), 'license', 'width'
    train_merged = pd.merge(
        images_df[["id", "file_name"]],
        annotations_df[["image_id", "category_id", "region_id"]],
        left_on="id",
        right_on="image_id",
        how="inner",
    )

    # Construct relative file paths
    # The json file_name is like "images/000/00/495523.jpg"
    # The physical path is "input/nybg2020/train/images/000/00/495523.jpg"
    # We want relative path from input dir: "nybg2020/train/images/000/00/495523.jpg"
    train_merged["file_path"] = "nybg2020/train/" + train_merged["file_name"]

    # Select necessary columns. 'image_id' is already present from annotations.
    train_merged = train_merged[["image_id", "file_path", "category_id", "region_id"]]

    # FIX: Drop duplicates based on image_id
    # Some images have multiple annotations. We keep the first one to ensure
    # each image appears only once in the dataset, preventing overlap after split.
    print("Checking for duplicate image_ids...")
    initial_len = len(train_merged)
    train_merged = train_merged.drop_duplicates(subset="image_id", keep="first")
    final_len = len(train_merged)
    if initial_len != final_len:
        print(
            f"Dropped {initial_len - final_len} duplicate annotations. Kept {final_len} unique images."
        )

    # Cite debug_lesson_1
    assert train_merged["image_id"].is_unique, "Duplicates remain after cleaning!"

    # --- Process Test Data ---
    print("Processing test data...")
    test_images_df = pd.DataFrame(test_data["images"])

    # Ensure IDs are integers
    test_images_df["id"] = test_images_df["id"].astype(int)

    # Construct relative file paths
    test_images_df["file_path"] = "nybg2020/test/" + test_images_df["file_name"]

    test_df = test_images_df[["id", "file_path"]].rename(columns={"id": "image_id"})

    # --- Split Train/Validation ---
    print("Splitting train/validation sets...")
    # Handle stratification for classes with too few samples
    # If a class has only 1 sample, it cannot be stratified into train and val.
    # We will put singletons in train.

    category_counts = train_merged["category_id"].value_counts()
    singletons = category_counts[category_counts < 2].index
    multiples = category_counts[category_counts >= 2].index

    df_singletons = train_merged[train_merged["category_id"].isin(singletons)]
    df_multiples = train_merged[train_merged["category_id"].isin(multiples)]

    # Stratified split on multiples
    train_split, val_split = train_test_split(
        df_multiples,
        test_size=0.2,
        stratify=df_multiples["category_id"],
        random_state=42,
    )

    # Combine singletons back into train
    train_final = (
        pd.concat([train_split, df_singletons])
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )
    val_final = val_split.sample(frac=1, random_state=42).reset_index(drop=True)

    # --- Save Metadata ---
    print("Saving metadata to CSV...")
    train_csv_path = os.path.join(METADATA_DIR, "train.csv")
    val_csv_path = os.path.join(METADATA_DIR, "val.csv")
    test_csv_path = os.path.join(METADATA_DIR, "test.csv")

    train_final.to_csv(train_csv_path, index=False)
    val_final.to_csv(val_csv_path, index=False)
    test_df.to_csv(test_csv_path, index=False)

    print(f"Saved {train_csv_path}")
    print(f"Saved {val_csv_path}")
    print(f"Saved {test_csv_path}")

    # --- Verification & Statistics ---
    print("\n==== Dataset Statistics ====")
    print(
        f"Training Set: {len(train_final)} images, {train_final['category_id'].nunique()} classes"
    )
    print(
        f"Validation Set: {len(val_final)} images, {val_final['category_id'].nunique()} classes"
    )
    print(f"Test Set: {len(test_df)} images")

    # Verify split logic
    train_ids = set(train_final["image_id"])
    val_ids = set(val_final["image_id"])
    intersection = train_ids.intersection(val_ids)
    if len(intersection) > 0:
        raise AssertionError(
            f"Found {len(intersection)} overlapping IDs between train and validation sets."
        )

    # Check stratification roughly (ratio of validation size)
    total_train_val = len(train_final) + len(val_final)
    val_ratio = len(val_final) / total_train_val
    print(f"Validation Ratio: {val_ratio:.4f} (Target ~0.2)")

    # --- File Path Verification ---
    print("\n==== Verifying File Paths ====")

    def check_paths(df, name):
        sample_size = min(1000, len(df))
        sample = df.sample(n=sample_size, random_state=42)
        missing_count = 0
        missing_samples = []

        for _, row in sample.iterrows():
            full_path = os.path.join(INPUT_DIR, row["file_path"])
            if not os.path.exists(full_path):
                missing_count += 1
                if len(missing_samples) < 5:
                    missing_samples.append(row["file_path"])

        missing_ratio = missing_count / sample_size
        print(
            f"[{name}] Missing files: {missing_count}/{sample_size} ({missing_ratio:.2%})"
        )

        if missing_count > 0:
            print(f"Sample missing paths in {name}:")
            for p in missing_samples:
                print(f"  - {p}")

        if missing_ratio > 0.5:
            raise FileNotFoundError(
                f"More than 50% of files missing in {name} dataset check."
            )

    check_paths(train_final, "Train")
    check_paths(val_final, "Validation")
    check_paths(test_df, "Test")

    print("\nMetadata generation and verification completed successfully.")


if __name__ == "__main__":
    main()
