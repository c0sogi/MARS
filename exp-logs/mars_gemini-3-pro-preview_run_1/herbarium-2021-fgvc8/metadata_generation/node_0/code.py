import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import random

# Configuration
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
TRAIN_META_PATH = os.path.join(INPUT_DIR, "train", "metadata.json")
TEST_META_PATH = os.path.join(INPUT_DIR, "test", "metadata.json")
RANDOM_STATE = 42
VAL_SIZE = 0.2


def main():
    # Create metadata directory
    os.makedirs(METADATA_DIR, exist_ok=True)

    print("Loading training metadata...")
    with open(TRAIN_META_PATH, "r") as f:
        train_meta = json.load(f)

    print("Processing training data...")
    # Create DataFrames from lists of dicts
    images_df = pd.DataFrame(train_meta["images"])
    annotations_df = pd.DataFrame(train_meta["annotations"])

    # Rename id in images to image_id for merging if necessary, or just map
    # images: [{'file_name': ..., 'height': ..., 'id': ..., ...}]
    # annotations: [{'category_id': ..., 'id': ..., 'image_id': ..., ...}]

    # Map image_id to file_name
    img_id_to_filename = pd.Series(
        images_df.file_name.values, index=images_df.id
    ).to_dict()

    # Prepare main dataframe
    df = annotations_df[["image_id", "category_id"]].copy()

    # Map file names
    df["file_name"] = df["image_id"].map(img_id_to_filename)

    # Construct relative path: train/images/...
    # The JSON file_name is like "images/000/00/123.jpg"
    df["file_path"] = "train/" + df["file_name"]

    # Drop rows where file_name might be missing (sanity check)
    if df["file_name"].isnull().any():
        print(
            f"Warning: {df['file_name'].isnull().sum()} annotations have no corresponding image info."
        )
        df = df.dropna(subset=["file_name"])

    # Ensure types
    df["image_id"] = df["image_id"].astype(int)
    df["category_id"] = df["category_id"].astype(int)

    # Stratified Split logic handling singletons
    print("Splitting train/val...")
    category_counts = df["category_id"].value_counts()
    singleton_categories = category_counts[category_counts < 2].index

    # Split data into singletons and rest
    singletons_df = df[df["category_id"].isin(singleton_categories)]
    rest_df = df[~df["category_id"].isin(singleton_categories)]

    print(f"Total samples: {len(df)}")
    print(f"Singleton samples (forced to train): {len(singletons_df)}")
    print(f"Stratifiable samples: {len(rest_df)}")

    # Stratified split on the rest
    train_rest, val_rest = train_test_split(
        rest_df,
        test_size=VAL_SIZE,
        stratify=rest_df["category_id"],
        random_state=RANDOM_STATE,
    )

    # Combine singletons with train
    train_final = (
        pd.concat([singletons_df, train_rest])
        .sample(frac=1, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )
    val_final = val_rest.reset_index(drop=True)

    print(f"Final Train size: {len(train_final)}")
    print(f"Final Val size: {len(val_final)}")

    # Save Train/Val
    train_csv_path = os.path.join(METADATA_DIR, "train.csv")
    val_csv_path = os.path.join(METADATA_DIR, "val.csv")

    # We only need specific columns
    cols_to_save = ["image_id", "file_path", "category_id"]
    train_final[cols_to_save].to_csv(train_csv_path, index=False)
    val_final[cols_to_save].to_csv(val_csv_path, index=False)

    # Clear memory
    del (
        train_meta,
        images_df,
        annotations_df,
        df,
        train_rest,
        val_rest,
        singletons_df,
        rest_df,
    )

    # Process Test
    print("Loading test metadata...")
    with open(TEST_META_PATH, "r") as f:
        test_meta = json.load(f)

    print("Processing test data...")
    test_images_df = pd.DataFrame(test_meta["images"])

    # Test metadata has 'id' and 'file_name'
    test_df = pd.DataFrame()
    test_df["image_id"] = test_images_df["id"].astype(int)
    test_df["file_name"] = test_images_df["file_name"]

    # Construct relative path: test/images/...
    test_df["file_path"] = "test/" + test_df["file_name"]

    test_csv_path = os.path.join(METADATA_DIR, "test.csv")
    test_df[["image_id", "file_path"]].to_csv(test_csv_path, index=False)

    print(f"Test size: {len(test_df)}")

    # --- Validation & Verification ---
    print("\nRunning validation checks...")

    # 1. Summary Statistics
    print("-" * 30)
    print("Summary Statistics:")
    print(f"Train samples: {len(train_final)}")
    print(f"Val samples: {len(val_final)}")
    print(f"Test samples: {len(test_df)}")
    print(f"Unique categories in Train: {train_final['category_id'].nunique()}")
    print(f"Unique categories in Val: {val_final['category_id'].nunique()}")
    print("-" * 30)

    # 2. File Existence Check
    def check_files(df, name):
        print(f"Checking file existence for {name} set...")
        sample_size = min(1000, len(df))
        sample = df.sample(n=sample_size, random_state=RANDOM_STATE)
        missing_count = 0
        missing_samples = []

        for _, row in sample.iterrows():
            full_path = os.path.join(INPUT_DIR, row["file_path"])
            if not os.path.exists(full_path):
                missing_count += 1
                if len(missing_samples) < 5:
                    missing_samples.append(row["file_path"])

        missing_ratio = missing_count / sample_size
        print(f"Missing file ratio for {name}: {missing_ratio:.4f}")

        if missing_ratio > 0.5:
            print("Sample missing files:")
            for p in missing_samples:
                print(p)
            raise FileNotFoundError(
                f"Too many missing files in {name} dataset! Ratio: {missing_ratio}"
            )

    check_files(train_final, "Train")
    check_files(val_final, "Val")
    check_files(test_df, "Test")

    # 3. Stratification Verification
    print("Verifying split requirements...")
    # Check that val has no categories that are not in train (basic requirement for valid split)
    train_cats = set(train_final["category_id"].unique())
    val_cats = set(val_final["category_id"].unique())

    if not val_cats.issubset(train_cats):
        diff = val_cats - train_cats
        raise AssertionError(
            f"Validation set contains categories not present in training set: {list(diff)[:5]}..."
        )

    # Check split ratio on stratifiable data
    # We expect val_final to be roughly 20% of (total - singletons)
    # Actually, val_final is exactly 20% of rest_df (rounded).
    # We can check if len(val_final) / (len(train_final) + len(val_final) - len(singletons)) is approx 0.2

    # A simpler check: Ensure we have roughly the right number of validation samples
    # We know we used test_size=0.2 on the non-singleton data.
    # Just asserting the code executed the split logic is mostly sufficient,
    # but let's check that the validation set is not empty and has expected size order of magnitude.
    assert len(val_final) > 0, "Validation set is empty!"

    # Check distribution consistency for a frequent class
    # Pick the most frequent class
    top_class = train_final["category_id"].mode()[0]
    train_density = (train_final["category_id"] == top_class).mean()
    val_density = (val_final["category_id"] == top_class).mean()

    print(f"Top class ID: {top_class}")
    print(f"Density in Train: {train_density:.5f}")
    print(f"Density in Val: {val_density:.5f}")

    # Allow small deviation due to singletons being only in train
    # The densities should be reasonably close.
    if abs(train_density - val_density) > 0.05:  # 5% absolute difference tolerance
        print(
            "Warning: Class distribution mismatch seems high. This might be due to many singletons."
        )
    else:
        print("Class distribution verification passed.")

    print("\nMetadata generation and validation complete.")


if __name__ == "__main__":
    main()
