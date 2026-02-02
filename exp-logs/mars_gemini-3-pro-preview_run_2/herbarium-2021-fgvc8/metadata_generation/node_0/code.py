import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Constants
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
RANDOM_STATE = 42
VAL_SIZE = 0.2


def main():
    # Ensure metadata directory exists
    os.makedirs(METADATA_DIR, exist_ok=True)

    print("Starting metadata generation...")

    # ---------------------------------------------------------
    # 1. Process Training Data
    # ---------------------------------------------------------
    train_json_path = os.path.join(INPUT_DIR, "train/metadata.json")
    print(f"Loading train metadata from {train_json_path}...")

    with open(train_json_path, "r") as f:
        train_data = json.load(f)

    print("Parsing train images and annotations...")
    # Create images dataframe
    df_train_imgs = pd.DataFrame(train_data["images"])
    # Ensure image_id is int and rename for consistency
    if "id" in df_train_imgs.columns:
        df_train_imgs.rename(columns={"id": "image_id"}, inplace=True)
    df_train_imgs["image_id"] = df_train_imgs["image_id"].astype(int)

    # Create annotations dataframe
    df_train_anns = pd.DataFrame(train_data["annotations"])
    # Ensure image_id is int
    df_train_anns["image_id"] = df_train_anns["image_id"].astype(int)

    # We only need image_id and category_id from annotations
    df_train_anns = df_train_anns[["image_id", "category_id"]]

    # Drop duplicate annotations for the same image if any (assuming single-label classification)
    df_train_anns = df_train_anns.drop_duplicates(subset=["image_id"])

    # Merge images and annotations
    print("Merging train data...")
    df_train = pd.merge(df_train_imgs, df_train_anns, on="image_id", how="inner")

    # Construct relative file path
    # The JSON file_name is like "images/000/00/xxxx.jpg"
    # The physical path is "input/train/images/000/00/xxxx.jpg"
    # We want the path relative to input, so "train/images/000/00/xxxx.jpg"
    df_train["file_path"] = "train/" + df_train["file_name"]

    # Select final columns
    df_train = df_train[["image_id", "file_path", "category_id"]]

    # Ensure category_id is int
    df_train["category_id"] = df_train["category_id"].astype(int)

    print(f"Total training samples found: {len(df_train)}")

    # ---------------------------------------------------------
    # 2. Split Training Data (Train/Val)
    # ---------------------------------------------------------
    print("Splitting training data into train and validation sets...")

    # Identify rare classes (count < 2) which cannot be stratified
    class_counts = df_train["category_id"].value_counts()
    rare_classes = class_counts[class_counts < 2].index

    # Separate rare and common classes
    df_rare = df_train[df_train["category_id"].isin(rare_classes)]
    df_common = df_train[~df_train["category_id"].isin(rare_classes)]

    print(f"Found {len(rare_classes)} rare classes with < 2 samples.")

    # Stratified split for common classes
    if len(df_common) > 0:
        train_common, val_common = train_test_split(
            df_common,
            test_size=VAL_SIZE,
            stratify=df_common["category_id"],
            random_state=RANDOM_STATE,
        )
    else:
        train_common = pd.DataFrame(columns=df_train.columns)
        val_common = pd.DataFrame(columns=df_train.columns)

    # Combine: rare classes go to train to ensure they are seen
    df_train_final = pd.concat([train_common, df_rare], axis=0)
    df_val_final = val_common.copy()

    # Shuffle
    df_train_final = df_train_final.sample(
        frac=1, random_state=RANDOM_STATE
    ).reset_index(drop=True)
    df_val_final = df_val_final.sample(frac=1, random_state=RANDOM_STATE).reset_index(
        drop=True
    )

    print(f"Final Train set size: {len(df_train_final)}")
    print(f"Final Validation set size: {len(df_val_final)}")

    # Save to CSV
    train_csv_path = os.path.join(METADATA_DIR, "train.csv")
    val_csv_path = os.path.join(METADATA_DIR, "val.csv")

    df_train_final.to_csv(train_csv_path, index=False)
    df_val_final.to_csv(val_csv_path, index=False)

    # Free memory
    del (
        train_data,
        df_train_imgs,
        df_train_anns,
        df_train,
        train_common,
        val_common,
        df_rare,
        df_common,
    )

    # ---------------------------------------------------------
    # 3. Process Test Data
    # ---------------------------------------------------------
    test_json_path = os.path.join(INPUT_DIR, "test/metadata.json")
    print(f"Loading test metadata from {test_json_path}...")

    with open(test_json_path, "r") as f:
        test_data = json.load(f)

    print("Processing test dataframe...")
    df_test = pd.DataFrame(test_data["images"])

    # Ensure image_id is int and rename
    if "id" in df_test.columns:
        df_test.rename(columns={"id": "image_id"}, inplace=True)
    df_test["image_id"] = df_test["image_id"].astype(int)

    # Construct relative file path
    # JSON file_name: "images/000/xxxx.jpg"
    # Target path: "test/images/000/xxxx.jpg"
    df_test["file_path"] = "test/" + df_test["file_name"]

    # Select columns
    df_test = df_test[["image_id", "file_path"]]

    print(f"Total test samples: {len(df_test)}")

    # Save to CSV
    test_csv_path = os.path.join(METADATA_DIR, "test.csv")
    df_test.to_csv(test_csv_path, index=False)

    del test_data, df_test

    # ---------------------------------------------------------
    # 4. Validation Checks
    # ---------------------------------------------------------
    print("\n--- Running Verification Checks ---")

    # Reload datasets
    df_train_check = pd.read_csv(train_csv_path)
    df_val_check = pd.read_csv(val_csv_path)
    df_test_check = pd.read_csv(test_csv_path)

    # 1. Summary Statistics
    print(f"Train samples: {len(df_train_check)}")
    print(f"Val samples: {len(df_val_check)}")
    print(f"Test samples: {len(df_test_check)}")
    print(f"Train unique classes: {df_train_check['category_id'].nunique()}")
    print(f"Val unique classes: {df_val_check['category_id'].nunique()}")

    # 2. File Existence Check
    def check_files(df, name):
        if len(df) == 0:
            print(f"Warning: {name} dataset is empty.")
            return

        print(f"Checking file existence for {name} set (sampling 1000 files)...")
        sample_size = min(1000, len(df))
        sample = df.sample(sample_size, random_state=RANDOM_STATE)

        missing_count = 0
        missing_examples = []

        for _, row in sample.iterrows():
            # Path in metadata is relative to ./input
            rel_path = row["file_path"]
            full_path = os.path.join(INPUT_DIR, rel_path)

            if not os.path.exists(full_path):
                missing_count += 1
                if len(missing_examples) < 5:
                    missing_examples.append(rel_path)

        ratio = missing_count / sample_size
        print(f"Missing file ratio for {name}: {ratio:.4f}")

        if missing_examples:
            print(f"Example missing files: {missing_examples}")

        if ratio > 0.5:
            raise FileNotFoundError(
                f"Verification failed: More than 50% of files are missing in the {name} dataset."
            )

    check_files(df_train_check, "Train")
    check_files(df_val_check, "Val")
    check_files(df_test_check, "Test")

    # 3. Stratification/Split Check
    if len(df_val_check) > 0:
        # Check that validation classes are present in training (except for potential edge cases, but with our logic they should be)
        val_classes = set(df_val_check["category_id"].unique())
        train_classes = set(df_train_check["category_id"].unique())

        if not val_classes.issubset(train_classes):
            print("Warning: Some classes in validation set are not in training set.")
            # This shouldn't happen with the current logic (rare classes -> train, common -> split)
            # unless a class had count >= 2 but split put all in val (unlikely with stratified split)
        else:
            print(
                "Class distribution check passed: All validation classes are present in training."
            )

    print("Metadata generation and verification completed successfully.")


if __name__ == "__main__":
    main()
