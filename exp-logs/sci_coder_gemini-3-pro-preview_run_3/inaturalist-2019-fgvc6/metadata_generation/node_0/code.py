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


def fix_file_paths(df, likely_parent_dir):
    """
    Checks if file paths in the DataFrame exist. If not, tries prepending
    the likely parent directory (e.g., 'train_val2019').
    """
    if df.empty:
        return df

    # Check the first file to determine path structure
    first_file = df.iloc[0]["file_name"]
    path_as_is = os.path.join(INPUT_DIR, first_file)

    if os.path.exists(path_as_is):
        return df

    # Try prepending the directory
    corrected_path = os.path.join(likely_parent_dir, first_file)
    full_corrected_path = os.path.join(INPUT_DIR, corrected_path)

    if os.path.exists(full_corrected_path):
        print(
            f"Detected missing parent directory in paths. Prepending '{likely_parent_dir}/'..."
        )
        df["file_name"] = df["file_name"].apply(
            lambda x: os.path.join(likely_parent_dir, x)
        )
        return df

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

    # Attempt to fix paths if the JSON 'file_name' doesn't include the extraction root
    # train and val images are expected to be in 'train_val2019'
    # test images are expected to be in 'test2019'
    train_df = fix_file_paths(train_df, "train_val2019")
    val_df = fix_file_paths(val_df, "train_val2019")
    test_df = fix_file_paths(test_df, "test2019")

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
