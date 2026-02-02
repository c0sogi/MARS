import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import gc


def main():
    # Define paths
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    TRAIN_JSON = os.path.join(INPUT_DIR, "train/metadata.json")
    TEST_JSON = os.path.join(INPUT_DIR, "test/metadata.json")

    # Create metadata directory
    os.makedirs(METADATA_DIR, exist_ok=True)

    print("Starting metadata generation...")

    # ---------------------------------------------------------
    # 1. Process Training Data
    # ---------------------------------------------------------
    print(f"Loading training metadata from {TRAIN_JSON}...")
    with open(TRAIN_JSON, "r") as f:
        train_meta = json.load(f)

    print("Parsing training images and annotations...")
    # Create a map from image_id to file_name
    # Ensure image_id is integer for consistency
    train_img_map = {int(img["id"]): img["file_name"] for img in train_meta["images"]}

    # Process annotations to create the dataset
    # We iterate over annotations to ensure we have labels
    train_data = []
    for ann in train_meta["annotations"]:
        img_id = int(ann["image_id"])
        category_id = int(ann["category_id"])

        if img_id in train_img_map:
            # Construct path relative to ./input
            # JSON file_name is like "images/000/..."
            # Physical path is input/train/images/000/...
            # So relative path is train/images/000/...
            file_path = os.path.join("train", train_img_map[img_id])

            train_data.append(
                {"image_id": img_id, "file_path": file_path, "category_id": category_id}
            )

    df_full = pd.DataFrame(train_data)
    print(f"Total labeled training samples: {len(df_full)}")

    # Clean up memory
    del train_meta, train_img_map, train_data
    gc.collect()

    # ---------------------------------------------------------
    # 2. Split Training Data (Train/Val)
    # ---------------------------------------------------------
    print("Splitting data into Train (80%) and Validation (20%)...")

    # Handle rare classes: Stratified split requires at least 2 samples per class
    class_counts = df_full["category_id"].value_counts()
    rare_classes = class_counts[class_counts < 2].index

    common_df = df_full[~df_full["category_id"].isin(rare_classes)]
    rare_df = df_full[df_full["category_id"].isin(rare_classes)]

    print(f"Classes with < 2 samples: {len(rare_classes)}")
    print(f"Samples in common classes: {len(common_df)}")
    print(f"Samples in rare classes (forced to train): {len(rare_df)}")

    # Perform stratified split on common classes
    train_common, val_common = train_test_split(
        common_df, test_size=0.2, stratify=common_df["category_id"], random_state=42
    )

    # Combine rare samples back into training set
    train_df = pd.concat([train_common, rare_df])
    val_df = val_common

    # Shuffle datasets
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    val_df = val_df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Final Train set size: {len(train_df)}")
    print(f"Final Val set size: {len(val_df)}")

    # Save to CSV
    train_csv_path = os.path.join(METADATA_DIR, "train.csv")
    val_csv_path = os.path.join(METADATA_DIR, "val.csv")
    train_df.to_csv(train_csv_path, index=False)
    val_df.to_csv(val_csv_path, index=False)
    print(f"Saved {train_csv_path} and {val_csv_path}")

    # Clean up
    del df_full, common_df, rare_df, train_common, val_common
    gc.collect()

    # ---------------------------------------------------------
    # 3. Process Test Data
    # ---------------------------------------------------------
    print(f"Loading test metadata from {TEST_JSON}...")
    with open(TEST_JSON, "r") as f:
        test_meta = json.load(f)

    print("Parsing test images...")
    test_data = []
    for img in test_meta["images"]:
        img_id = int(img["id"])
        # Construct path relative to ./input
        # Physical path is input/test/images/...
        file_path = os.path.join("test", img["file_name"])

        test_data.append({"image_id": img_id, "file_path": file_path})

    df_test = pd.DataFrame(test_data)
    # Sort by image_id for neatness
    df_test = df_test.sort_values("image_id").reset_index(drop=True)

    print(f"Total test samples: {len(df_test)}")

    # Save to CSV
    test_csv_path = os.path.join(METADATA_DIR, "test.csv")
    df_test.to_csv(test_csv_path, index=False)
    print(f"Saved {test_csv_path}")

    del test_meta, test_data
    gc.collect()

    # ---------------------------------------------------------
    # 4. Verification
    # ---------------------------------------------------------
    print("\n" + "=" * 30)
    print("VERIFICATION CHECKS")
    print("=" * 30)

    # A. Summary Statistics
    datasets = [("Train", train_df), ("Val", val_df), ("Test", df_test)]
    for name, df in datasets:
        print(f"\n[{name} Dataset]")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        if "category_id" in df.columns:
            n_classes = df["category_id"].nunique()
            print(f"Unique Categories: {n_classes}")

    # B. File Existence Check
    print("\n[File Existence Check]")
    for name, df in datasets:
        print(f"Checking random sample of 1000 files from {name}...")
        if len(df) == 0:
            continue

        sample = df.sample(n=min(1000, len(df)), random_state=42)
        missing_count = 0
        missing_samples = []

        for _, row in sample.iterrows():
            # file_path is relative to ./input
            full_path = os.path.join(INPUT_DIR, row["file_path"])
            if not os.path.exists(full_path):
                missing_count += 1
                if len(missing_samples) < 5:
                    missing_samples.append(row["file_path"])

        ratio = missing_count / len(sample)
        print(f"  Missing Ratio: {ratio:.4f} ({missing_count}/{len(sample)})")

        if missing_count > 0:
            print(f"  Sample missing files: {missing_samples}")

        if ratio > 0.5:
            raise FileNotFoundError(
                f"CRITICAL: More than 50% of files are missing in {name} dataset!"
            )

    # C. Split Verification
    print("\n[Split Verification]")
    train_ids = set(train_df["image_id"])
    val_ids = set(val_df["image_id"])

    # Check for overlap
    overlap = train_ids.intersection(val_ids)
    if overlap:
        raise AssertionError(
            f"CRITICAL: Found {len(overlap)} overlapping IDs between Train and Val sets!"
        )
    else:
        print("  No overlap between Train and Val sets.")

    # Check ratio
    total_train_val = len(train_df) + len(val_df)
    val_ratio = len(val_df) / total_train_val
    print(f"  Validation Split Ratio: {val_ratio:.4f} (Target: ~0.20)")

    # We allow some deviation because rare classes were forced into train
    if not (0.15 <= val_ratio <= 0.25):
        print(
            "  WARNING: Validation ratio deviates from 0.20, likely due to rare class handling."
        )
    else:
        print("  Split ratio is within acceptable range.")

    print("\nMetadata generation completed successfully.")


if __name__ == "__main__":
    main()
