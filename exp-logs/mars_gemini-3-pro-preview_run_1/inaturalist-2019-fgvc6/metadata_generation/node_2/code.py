import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# Constants
INPUT_DIR = Path("./input")
METADATA_DIR = Path("./metadata")
RANDOM_STATE = 42


def load_json_dataset(json_path, has_annotations=True):
    """Loads a COCO-style JSON dataset and returns a DataFrame."""
    if not json_path.exists():
        return pd.DataFrame()

    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Failed to load {json_path}: {e}")
        return pd.DataFrame()

    if "images" not in data:
        return pd.DataFrame()

    images_df = pd.DataFrame(data["images"])
    # Ensure required columns exist
    if "id" not in images_df.columns or "file_name" not in images_df.columns:
        return pd.DataFrame()

    if has_annotations:
        if "annotations" not in data:
            return pd.DataFrame()
        annotations_df = pd.DataFrame(data["annotations"])

        # Merge images and annotations
        # images: id, file_name
        # annotations: image_id, category_id
        merged_df = pd.merge(
            images_df[["id", "file_name"]],
            annotations_df[["image_id", "category_id"]],
            left_on="id",
            right_on="image_id",
            how="inner",
        )
        # Rename and select columns
        merged_df = merged_df.rename(
            columns={"id": "image_id", "file_name": "file_path"}
        )
        return merged_df[["image_id", "file_path", "category_id"]]
    else:
        # Just images
        images_df = images_df.rename(
            columns={"id": "image_id", "file_name": "file_path"}
        )
        return images_df[["image_id", "file_path"]]


def verify_files(df, name):
    """Checks if files exist in the input directory. Returns True if pass, False if fail."""
    if df.empty:
        print(f"Skipping file verification for empty dataset: {name}")
        return True

    sample_size = min(1000, len(df))
    sample = df.sample(n=sample_size, random_state=RANDOM_STATE)
    missing_count = 0
    missing_samples = []

    for _, row in sample.iterrows():
        file_path = INPUT_DIR / row["file_path"]
        if not file_path.exists():
            missing_count += 1
            if len(missing_samples) < 5:
                missing_samples.append(str(file_path))

    missing_ratio = missing_count / sample_size
    print(
        f"[{name}] Missing file ratio: {missing_ratio:.4f} ({missing_count}/{sample_size})"
    )

    if missing_ratio > 0.5:
        print(f"[{name}] Sample missing files: {missing_samples}")
        return False

    return True


def main():
    METADATA_DIR.mkdir(exist_ok=True)

    print("Loading datasets...")
    train_df = load_json_dataset(INPUT_DIR / "train2019.json", has_annotations=True)
    val_df = load_json_dataset(INPUT_DIR / "val2019.json", has_annotations=True)
    test_df = load_json_dataset(INPUT_DIR / "test2019.json", has_annotations=False)

    print(
        f"Initial counts - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}"
    )

    # Verify Train Files
    print("Verifying Train files...")
    if not verify_files(train_df, "Train"):
        raise FileNotFoundError("Critical: Training files missing!")

    # Verify Validation Files
    print("Verifying Validation files...")
    if not verify_files(val_df, "Validation"):
        print("Validation files are missing. Discarding provided validation set.")
        val_df = pd.DataFrame()

    # Handle Validation Split
    created_split = False
    if val_df.empty:
        print(
            "Validation set is empty or missing. Creating split from training data..."
        )
        if train_df.empty:
            raise ValueError("Training data is also empty. Cannot proceed.")

        # Cite debug_lesson_1: Split Unique Entities, Not Rows
        unique_imgs = train_df.drop_duplicates(subset=["image_id"])

        try:
            train_imgs, val_imgs = train_test_split(
                unique_imgs,
                test_size=0.2,
                random_state=RANDOM_STATE,
                stratify=unique_imgs["category_id"],
            )
            print("Performed stratified split on unique image IDs.")
        except ValueError as e:
            print(
                f"Stratified split failed ({e}). Falling back to random split on IDs."
            )
            train_imgs, val_imgs = train_test_split(
                unique_imgs, test_size=0.2, random_state=RANDOM_STATE
            )

        train_split = train_df[train_df["image_id"].isin(train_imgs["image_id"])]
        val_split = train_df[train_df["image_id"].isin(val_imgs["image_id"])]

        train_df = train_split
        val_df = val_split
        created_split = True

    # Save Metadata
    print("Saving metadata CSVs...")
    train_df.to_csv(METADATA_DIR / "train.csv", index=False)
    val_df.to_csv(METADATA_DIR / "val.csv", index=False)
    test_df.to_csv(METADATA_DIR / "test.csv", index=False)

    # Final Statistics and Verification
    print("\n==== Dataset Statistics ====")
    datasets = {"Train": train_df, "Validation": val_df, "Test": test_df}

    for name, df in datasets.items():
        print(f"--- {name} ---")
        print(f"Total samples: {len(df)}")
        if "category_id" in df.columns:
            print(f"Unique classes: {df['category_id'].nunique()}")
            if not df.empty:
                print(
                    f"Class distribution head:\n{df['category_id'].value_counts().head()}"
                )

    if created_split:
        print("\nVerifying Split Requirements...")
        total_samples = len(train_df) + len(val_df)
        val_ratio = len(val_df) / total_samples
        print(f"Validation Ratio: {val_ratio:.4f}")

        # Check ratio (approx 0.2)
        if not (0.19 <= val_ratio <= 0.21):
            raise AssertionError(
                f"Validation split ratio {val_ratio:.4f} is not close to 0.2"
            )

        # Check overlap
        train_ids = set(train_df["image_id"])
        val_ids = set(val_df["image_id"])
        overlap = train_ids.intersection(val_ids)
        if overlap:
            raise AssertionError(
                f"Found {len(overlap)} overlapping images between train and val sets."
            )
        print("Split verification passed.")

    print("\nMetadata generation complete.")


if __name__ == "__main__":
    main()
