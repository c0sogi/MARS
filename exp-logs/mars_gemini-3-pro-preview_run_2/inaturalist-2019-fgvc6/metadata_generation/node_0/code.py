import json
import pandas as pd
import os
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split


def generate_metadata():
    INPUT_DIR = Path("./input")
    METADATA_DIR = Path("./metadata")
    METADATA_DIR.mkdir(exist_ok=True, parents=True)

    # --- Helper Function to Process COCO JSON ---
    def process_coco_json(json_file, has_annotations=True):
        if not json_file.exists():
            return pd.DataFrame()

        print(f"Loading {json_file}...")
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Failed to load {json_file}: {e}")
            return pd.DataFrame()

        if "images" not in data or not data["images"]:
            return pd.DataFrame()

        images_df = pd.DataFrame(data["images"])
        # Ensure image_id column exists
        if "id" in images_df.columns:
            images_df = images_df.rename(columns={"id": "image_id"})

        merged = images_df

        if has_annotations and "annotations" in data:
            anns_df = pd.DataFrame(data["annotations"])
            if not anns_df.empty:
                # anns_df has 'id' (annotation id), 'image_id', 'category_id'
                # Merge on image_id
                merged = pd.merge(images_df, anns_df, on="image_id", how="left")
                # Drop rows where category_id is NaN (images without annotations)
                merged = merged.dropna(subset=["category_id"])
                merged["category_id"] = merged["category_id"].astype(int)
            else:
                # Annotations key exists but is empty
                return pd.DataFrame()
        elif has_annotations:
            # Expected annotations but none found in JSON structure
            return pd.DataFrame()

        # Add category names if available
        if "categories" in data:
            cats_df = pd.DataFrame(data["categories"])
            if not cats_df.empty:
                cats_df = cats_df.rename(
                    columns={"id": "category_id", "name": "category_name"}
                )
                if "category_id" in merged.columns:
                    merged = pd.merge(
                        merged,
                        cats_df[["category_id", "category_name"]],
                        on="category_id",
                        how="left",
                    )

        return merged

    # --- 1. Load Datasets ---
    print("Processing training data...")
    train_df = process_coco_json(INPUT_DIR / "train2019.json", has_annotations=True)

    print("Processing validation data...")
    val_df = process_coco_json(INPUT_DIR / "val2019.json", has_annotations=True)

    print("Processing test data...")
    test_df = process_coco_json(INPUT_DIR / "test2019.json", has_annotations=False)

    # --- 2. Validation Split Logic ---
    # If val_df is empty or missing annotations, split train_df
    if val_df.empty:
        print("Validation set not found or empty. Creating split from training set...")

        if train_df.empty:
            raise ValueError("Training data is empty. Cannot proceed.")

        # Handle rare classes that appear less than 2 times (cannot be stratified)
        class_counts = train_df["category_id"].value_counts()
        rare_classes = class_counts[class_counts < 2].index

        rare_df = train_df[train_df["category_id"].isin(rare_classes)]
        common_df = train_df[~train_df["category_id"].isin(rare_classes)]

        print(
            f"Splitting {len(common_df)} samples. Keeping {len(rare_df)} rare samples in train."
        )

        train_split, val_split = train_test_split(
            common_df, test_size=0.2, random_state=42, stratify=common_df["category_id"]
        )

        # Combine split train with rare classes
        train_df = pd.concat([train_split, rare_df])
        val_df = val_split
        print(f"New split - Train: {len(train_df)}, Val: {len(val_df)}")
    else:
        print(
            f"Using provided validation set. Train: {len(train_df)}, Val: {len(val_df)}"
        )

    # --- 3. Path Correction ---
    # Check if file paths need a prefix (e.g. 'train_val2019/')
    def fix_paths(df, dataset_name):
        if df.empty:
            return df

        sample_file = df.iloc[0]["file_name"]
        current_path = INPUT_DIR / sample_file

        # If path exists as is, no change needed
        if current_path.exists():
            return df

        # Check if prepending 'train_val2019/' helps
        # This is the most likely folder for train/val images
        alt_path = INPUT_DIR / "train_val2019" / sample_file
        if alt_path.exists():
            print(f"Prepending 'train_val2019/' to {dataset_name} file paths.")
            df["file_name"] = "train_val2019/" + df["file_name"]
            return df

        # Check if prepending 'test2019/' helps (unlikely if json is correct, but good for safety)
        alt_path_test = INPUT_DIR / "test2019" / sample_file
        if alt_path_test.exists():
            print(f"Prepending 'test2019/' to {dataset_name} file paths.")
            df["file_name"] = "test2019/" + df["file_name"]
            return df

        print(
            f"Warning: Could not resolve path for sample {sample_file} in {dataset_name}."
        )
        return df

    train_df = fix_paths(train_df, "train")
    val_df = fix_paths(val_df, "val")
    test_df = fix_paths(test_df, "test")

    # --- 4. Save Metadata ---
    common_cols = ["image_id", "file_name", "category_id"]
    if "category_name" in train_df.columns:
        common_cols.append("category_name")

    print("Saving metadata...")
    train_df[common_cols].to_csv(METADATA_DIR / "train.csv", index=False)
    val_df[common_cols].to_csv(METADATA_DIR / "val.csv", index=False)

    test_cols = ["image_id", "file_name"]
    test_df[test_cols].to_csv(METADATA_DIR / "test.csv", index=False)

    # --- 5. Verification & Statistics ---
    print("\n=== Dataset Statistics ===")
    for name, path in [
        ("Train", METADATA_DIR / "train.csv"),
        ("Val", METADATA_DIR / "val.csv"),
        ("Test", METADATA_DIR / "test.csv"),
    ]:
        df = pd.read_csv(path)
        print(f"\nDataset: {name}")
        print(f"Total samples: {len(df)}")
        if "category_id" in df.columns:
            print(f"Number of classes: {df['category_id'].nunique()}")
            print("Class distribution (top 5):")
            print(df["category_id"].value_counts().head(5))

        # Path Verification
        print(f"Verifying file paths for {name}...")
        sample_size = 1000
        if len(df) > sample_size:
            sample_df = df.sample(sample_size, random_state=42)
        else:
            sample_df = df

        missing_count = 0
        missing_examples = []

        for _, row in sample_df.iterrows():
            file_path = INPUT_DIR / row["file_name"]
            if not file_path.exists():
                missing_count += 1
                if len(missing_examples) < 5:
                    missing_examples.append(str(file_path))

        missing_ratio = missing_count / len(sample_df)
        print(f"Missing file ratio: {missing_ratio:.4f}")

        if missing_ratio > 0.5:
            print("Examples of missing files:")
            for m in missing_examples:
                print(m)
            raise FileNotFoundError(
                f"Missing file ratio ({missing_ratio:.2f}) is too high for {name} dataset."
            )

    print("\nMetadata generation and verification completed successfully.")


if __name__ == "__main__":
    generate_metadata()
