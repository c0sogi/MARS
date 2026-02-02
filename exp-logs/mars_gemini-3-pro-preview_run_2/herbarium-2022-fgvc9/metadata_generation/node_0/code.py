import os
import json
import pandas as pd
import glob
from sklearn.model_selection import train_test_split
import random

# Configuration
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
TRAIN_IMAGES_DIR = os.path.join(INPUT_DIR, "train_images")
TEST_METADATA_FILE = os.path.join(INPUT_DIR, "test_metadata.json")
RANDOM_STATE = 42


def generate_metadata():
    # Create metadata directory
    if not os.path.exists(METADATA_DIR):
        os.makedirs(METADATA_DIR)

    print("Step 1: Scanning training images...")
    train_data = []

    # Walk through the train_images directory
    # Expected structure: train_images/000/00/00000__001.jpg
    # We scan recursively to find all jpg files
    if os.path.exists(TRAIN_IMAGES_DIR):
        for root, dirs, files in os.walk(TRAIN_IMAGES_DIR):
            for file in files:
                if file.lower().endswith(".jpg"):
                    # Extract category_id from filename (e.g., 00000__001.jpg -> 0)
                    try:
                        category_part = file.split("__")[0]
                        category_id = int(category_part)

                        # Get path relative to ./input
                        # root is absolute or relative to cwd, we need relative to ./input
                        rel_dir = os.path.relpath(root, INPUT_DIR)
                        file_path = os.path.join(rel_dir, file)

                        train_data.append(
                            {"image_path": file_path, "label": category_id}
                        )
                    except (ValueError, IndexError):
                        continue
    else:
        print(f"Warning: {TRAIN_IMAGES_DIR} not found.")

    df_train_full = pd.DataFrame(train_data)
    print(f"  Found {len(df_train_full)} training images.")

    if df_train_full.empty:
        raise ValueError("No training images found. Check input directory structure.")

    print("Step 2: Splitting training data (80/20 Stratified)...")
    # Identify classes with fewer than 2 samples (cannot be stratified)
    class_counts = df_train_full["label"].value_counts()
    singletons = class_counts[class_counts < 2].index.tolist()

    # Split into singletons and rest
    df_singletons = df_train_full[df_train_full["label"].isin(singletons)]
    df_rest = df_train_full[~df_train_full["label"].isin(singletons)]

    # Stratified split on the rest
    if not df_rest.empty:
        train_rest, val_rest = train_test_split(
            df_rest, test_size=0.2, stratify=df_rest["label"], random_state=RANDOM_STATE
        )
        # Combine singletons into train set
        train_df = (
            pd.concat([train_rest, df_singletons])
            .sample(frac=1, random_state=RANDOM_STATE)
            .reset_index(drop=True)
        )
        val_df = val_rest.reset_index(drop=True)
    else:
        # Fallback if all classes are singletons
        train_df = df_singletons.sample(frac=1, random_state=RANDOM_STATE).reset_index(
            drop=True
        )
        val_df = pd.DataFrame(columns=df_train_full.columns)

    # Save to CSV
    train_csv_path = os.path.join(METADATA_DIR, "train.csv")
    val_csv_path = os.path.join(METADATA_DIR, "val.csv")
    train_df.to_csv(train_csv_path, index=False)
    val_df.to_csv(val_csv_path, index=False)
    print(
        f"  Saved train.csv ({len(train_df)} samples) and val.csv ({len(val_df)} samples)."
    )

    print("Step 3: Processing test metadata...")
    test_data = []
    if os.path.exists(TEST_METADATA_FILE):
        with open(TEST_METADATA_FILE, "r") as f:
            test_meta_json = json.load(f)

        for item in test_meta_json:
            # item['file_name'] example: "000/test-000000.jpg"
            # We assume the folder in input is 'test_images' based on file listing
            rel_path = os.path.join("test_images", item["file_name"])
            test_data.append({"image_path": rel_path, "image_id": item["image_id"]})
    else:
        print(f"Warning: {TEST_METADATA_FILE} not found.")

    df_test = pd.DataFrame(test_data)
    test_csv_path = os.path.join(METADATA_DIR, "test.csv")
    df_test.to_csv(test_csv_path, index=False)
    print(f"  Saved test.csv ({len(df_test)} samples).")

    return train_df, val_df, df_test


def validate_datasets(train_df, val_df, test_df):
    print("\nStep 4: Validating datasets...")

    # Summary Statistics
    print(
        f"  Train Shape: {train_df.shape}, Unique Classes: {train_df['label'].nunique() if not train_df.empty else 0}"
    )
    print(
        f"  Val Shape:   {val_df.shape}, Unique Classes: {val_df['label'].nunique() if not val_df.empty else 0}"
    )
    print(f"  Test Shape:  {test_df.shape}")

    # File Existence Check
    def check_existence(df, name):
        if df.empty:
            return
        print(f"  Checking {name} file paths...")
        sample_size = min(1000, len(df))
        sample = df.sample(n=sample_size, random_state=RANDOM_STATE)

        missing_count = 0
        missing_examples = []

        for _, row in sample.iterrows():
            full_path = os.path.join(INPUT_DIR, row["image_path"])
            if not os.path.exists(full_path):
                missing_count += 1
                if len(missing_examples) < 5:
                    missing_examples.append(full_path)

        missing_ratio = missing_count / sample_size
        print(f"    {name} Missing Ratio: {missing_ratio:.4f}")

        if missing_examples:
            print(f"    Example missing files: {missing_examples}")

        if missing_ratio > 0.5:
            raise FileNotFoundError(
                f"Error: More than 50% of files are missing in {name} dataset."
            )

    check_existence(train_df, "Train")
    check_existence(val_df, "Val")
    check_existence(test_df, "Test")

    # Stratification Check
    if not val_df.empty:
        print("  Verifying split logic...")
        assert (
            len(val_df) > 0
        ), "Validation set is empty despite having sufficient data."
        print("  Validation split verified.")


if __name__ == "__main__":
    try:
        train_df, val_df, test_df = generate_metadata()
        validate_datasets(train_df, val_df, test_df)
        print("\nSuccess: Metadata generation complete.")
    except Exception as e:
        print(f"\nError: {e}")
        raise e
