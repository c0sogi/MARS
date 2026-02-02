import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Configuration
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
RANDOM_STATE = 42


def load_json(file_path):
    """Loads a JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)


def create_df_from_coco(json_data, is_test=False):
    """Parses COCO-format JSON data into a DataFrame."""
    images = pd.DataFrame(json_data["images"])

    # Normalize column names
    if "id" in images.columns and "image_id" not in images.columns:
        images.rename(columns={"id": "image_id"}, inplace=True)

    # Select relevant columns
    df = images[["image_id", "file_name"]].copy()

    if not is_test:
        if "annotations" in json_data:
            annotations = pd.DataFrame(json_data["annotations"])
            if not annotations.empty:
                # Keep only image_id and category_id
                if (
                    "image_id" in annotations.columns
                    and "category_id" in annotations.columns
                ):
                    # Drop duplicates to ensure one label per image (if multiple exist, take first/unique)
                    # iNaturalist is typically single-label
                    anns = annotations[["image_id", "category_id"]].drop_duplicates(
                        subset=["image_id"]
                    )
                    df = pd.merge(df, anns, on="image_id", how="left")

    return df


def build_file_map(root_dirs):
    """
    Scans directories recursively and returns a dict: basename -> relative_path
    relative_path is relative to INPUT_DIR.
    """
    file_map = {}
    for root_dir in root_dirs:
        print(f"Scanning {root_dir}...")
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                full_path = os.path.join(root, file)
                # We want the path relative to INPUT_DIR for the dataframe
                rel_path = os.path.relpath(full_path, INPUT_DIR)
                file_map[file] = rel_path
    print(f"Mapped {len(file_map)} files.")
    return file_map


def fix_paths_and_labels(df, file_map):
    """
    Updates file_name based on file_map.
    Also updates category_id based on the directory structure if possible.
    """
    if df.empty:
        return df

    print("Fixing paths and labels...")

    # Create a basename column for mapping
    df["basename"] = df["file_name"].apply(os.path.basename)

    # Map to new paths
    # If basename not in map, keep original file_name (via fillna)
    df["new_path"] = df["basename"].map(file_map)
    df["file_name"] = df["new_path"].fillna(df["file_name"])

    # Update category_id if present
    if "category_id" in df.columns:
        # Extract category from path: .../{Category}/{Image}
        # We split by os.sep and take the second to last element
        def extract_category(path):
            parts = path.split(os.sep)
            if len(parts) >= 2:
                # Check if second to last is digits (the category ID)
                if parts[-2].isdigit():
                    return int(parts[-2])
            return None

        # Apply extraction
        extracted_cats = df["file_name"].apply(extract_category)

        # Update category_id where extraction was successful
        df["category_id"] = extracted_cats.fillna(df["category_id"])

    df.drop(columns=["basename", "new_path"], inplace=True)
    return df


def validate_dataset(df, name):
    """
    Validates that a sample of file paths exist. Raises error if >50% missing.
    """
    if df.empty:
        print(f"Warning: {name} dataset is empty.")
        return

    sample_size = min(1000, len(df))
    sample = df.sample(n=sample_size, random_state=RANDOM_STATE)

    missing_paths = []
    for _, row in sample.iterrows():
        full_path = os.path.join(INPUT_DIR, row["file_name"])
        if not os.path.exists(full_path):
            missing_paths.append(row["file_name"])

    missing_ratio = len(missing_paths) / sample_size
    print(f"{name} missing file ratio: {missing_ratio:.4f}")

    if missing_ratio > 0.5:
        print(f"Sample missing paths from {name}:")
        for p in missing_paths[:5]:
            print(f" - {p}")
        raise FileNotFoundError(
            f"More than 50% of files in {name} dataset are missing."
        )


def main():
    os.makedirs(METADATA_DIR, exist_ok=True)

    print("Loading JSON data...")
    train_data = load_json(os.path.join(INPUT_DIR, "train2019.json"))
    val_data = load_json(os.path.join(INPUT_DIR, "val2019.json"))
    test_data = load_json(os.path.join(INPUT_DIR, "test2019.json"))

    print("Processing DataFrames...")
    train_df = create_df_from_coco(train_data, is_test=False)
    val_df = create_df_from_coco(val_data, is_test=False)
    test_df = create_df_from_coco(test_data, is_test=True)

    # Build file map to resolve path mismatches
    # We scan both train_val2019 and test2019
    train_val_dir = os.path.join(INPUT_DIR, "train_val2019")
    test_dir = os.path.join(INPUT_DIR, "test2019")

    dirs_to_scan = []
    if os.path.exists(train_val_dir):
        dirs_to_scan.append(train_val_dir)
    if os.path.exists(test_dir):
        dirs_to_scan.append(test_dir)

    file_map = build_file_map(dirs_to_scan)

    # Fix paths and labels
    print("Updating Train DataFrame...")
    train_df = fix_paths_and_labels(train_df, file_map)
    print("Updating Val DataFrame...")
    val_df = fix_paths_and_labels(val_df, file_map)
    print("Updating Test DataFrame...")
    test_df = fix_paths_and_labels(test_df, file_map)

    # Handle Validation Split
    if len(val_df) == 0:
        print("Validation set provided is empty. Creating split from training data...")

        # Check for classes with too few samples to stratify
        class_counts = train_df["category_id"].value_counts()
        singletons = class_counts[class_counts < 2].index

        if len(singletons) > 0:
            print(
                f"Warning: {len(singletons)} classes have < 2 samples. Excluding them from validation split."
            )
            mask_singletons = train_df["category_id"].isin(singletons)
            singletons_df = train_df[mask_singletons]
            rest_df = train_df[~mask_singletons]

            train_split, val_split = train_test_split(
                rest_df,
                test_size=0.2,
                stratify=rest_df["category_id"],
                random_state=RANDOM_STATE,
            )

            # Add singletons back to train
            train_df = pd.concat([train_split, singletons_df])
            val_df = val_split
        else:
            train_df, val_df = train_test_split(
                train_df,
                test_size=0.2,
                stratify=train_df["category_id"],
                random_state=RANDOM_STATE,
            )

        # Verify split
        if len(val_df) == 0:
            raise AssertionError("Validation split failed: Validation set is empty.")

        # Verify stratification overlap (basic check)
        train_classes = set(train_df["category_id"].unique())
        val_classes = set(val_df["category_id"].unique())
        common = train_classes.intersection(val_classes)
        if len(common) == 0:
            raise AssertionError(
                "Validation split failed: No class overlap between train and val."
            )
        print("Validation split verified.")

    # Save Metadata
    print("Saving metadata to disk...")
    train_df.to_csv(os.path.join(METADATA_DIR, "train_metadata.csv"), index=False)
    val_df.to_csv(os.path.join(METADATA_DIR, "val_metadata.csv"), index=False)
    test_df.to_csv(os.path.join(METADATA_DIR, "test_metadata.csv"), index=False)

    # Print Statistics
    print("\n=== Dataset Statistics ===")
    print(f"Train samples: {len(train_df)}")
    print(f"Val samples:   {len(val_df)}")
    print(f"Test samples:  {len(test_df)}")
    if "category_id" in train_df.columns:
        print(f"Train classes: {train_df['category_id'].nunique()}")
    if "category_id" in val_df.columns:
        print(f"Val classes:   {val_df['category_id'].nunique()}")

    # Validate File Existence
    print("\n=== Validating File Paths ===")
    validate_dataset(train_df, "Train")
    validate_dataset(val_df, "Validation")
    validate_dataset(test_df, "Test")

    print("\nMetadata generation and validation complete.")


if __name__ == "__main__":
    main()
