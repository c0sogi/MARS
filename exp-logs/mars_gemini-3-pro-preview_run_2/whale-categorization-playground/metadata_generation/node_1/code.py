import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def main():
    # Configuration
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    RANDOM_STATE = 42
    VAL_SIZE = 0.2

    # Ensure metadata directory exists
    if not os.path.exists(METADATA_DIR):
        os.makedirs(METADATA_DIR)

    print("Starting metadata generation...")

    # ---------------------------------------------------------
    # 1. Process Training Data & Create Validation Split
    # ---------------------------------------------------------
    train_csv_path = os.path.join(INPUT_DIR, "train.csv")
    if not os.path.exists(train_csv_path):
        raise FileNotFoundError(f"train.csv not found at {train_csv_path}")

    train_df_raw = pd.read_csv(train_csv_path)

    # Construct relative file paths (Images are in 'train/' subdirectory)
    train_df_raw["file_path"] = train_df_raw["Image"].apply(
        lambda x: os.path.join("train", x)
    )

    # Handle Stratification with Class Imbalance
    # Strategy:
    # 1. Classes with < 5 samples cannot be stratified (would imply 0 in one set).
    #    These must go to Training to be learnable.
    # 2. Classes with >= 5 samples are split 80/20 stratified.

    class_counts = train_df_raw["Id"].value_counts()
    singletons = class_counts[class_counts < 5].index
    multi_samples = class_counts[class_counts >= 5].index

    df_singletons = train_df_raw[train_df_raw["Id"].isin(singletons)]
    df_multi = train_df_raw[train_df_raw["Id"].isin(multi_samples)]

    # Perform stratified split on multi-sample classes
    train_multi, val_multi = train_test_split(
        df_multi,
        test_size=VAL_SIZE,
        stratify=df_multi["Id"],
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    # Combine singletons into training set
    train_final = pd.concat([train_multi, df_singletons], axis=0)
    # Shuffle training set to mix singletons and multi-samples
    train_final = train_final.sample(frac=1, random_state=RANDOM_STATE).reset_index(
        drop=True
    )
    val_final = val_multi.reset_index(drop=True)

    # Save to metadata
    train_meta_path = os.path.join(METADATA_DIR, "train.csv")
    val_meta_path = os.path.join(METADATA_DIR, "val.csv")

    train_final.to_csv(train_meta_path, index=False)
    val_final.to_csv(val_meta_path, index=False)

    # ---------------------------------------------------------
    # 2. Process Test Data
    # ---------------------------------------------------------
    test_dir = os.path.join(INPUT_DIR, "test")
    # Use glob to find all jpg files
    test_files = glob.glob(os.path.join(test_dir, "*.jpg"))

    test_data = []
    for p in test_files:
        filename = os.path.basename(p)
        # Path relative to ./input
        rel_path = os.path.join("test", filename)
        test_data.append({"Image": filename, "file_path": rel_path, "Id": None})

    test_df = pd.DataFrame(test_data)
    test_meta_path = os.path.join(METADATA_DIR, "test.csv")
    test_df.to_csv(test_meta_path, index=False)

    print("Metadata files generated successfully.")

    # ---------------------------------------------------------
    # 3. Verification and Statistics
    # ---------------------------------------------------------
    print("\n=== Dataset Verification & Statistics ===")

    # Load generated metadata
    meta_train = pd.read_csv(train_meta_path)
    meta_val = pd.read_csv(val_meta_path)
    meta_test = pd.read_csv(test_meta_path)

    # Summary Statistics
    print(f"Train Samples: {len(meta_train)}")
    print(f"Val Samples:   {len(meta_val)}")
    print(f"Test Samples:  {len(meta_test)}")
    print(f"Train Unique IDs: {meta_train['Id'].nunique()}")
    print(f"Val Unique IDs:   {meta_val['Id'].nunique()}")

    # 1. Check File Paths (Missing File Ratio)
    def verify_files(df, name):
        if len(df) == 0:
            print(f"Warning: {name} dataset is empty.")
            return

        # Check up to 1000 random files
        sample_size = min(1000, len(df))
        sample = df.sample(n=sample_size, random_state=RANDOM_STATE)

        missing_count = 0
        missing_examples = []

        for _, row in sample.iterrows():
            # Construct full path from relative path
            full_path = os.path.join(INPUT_DIR, row["file_path"])
            if not os.path.exists(full_path):
                missing_count += 1
                if len(missing_examples) < 5:
                    missing_examples.append(row["file_path"])

        ratio = missing_count / sample_size
        print(f"Missing file ratio for {name}: {ratio:.4f}")

        if ratio > 0.5:
            print(f"Sample missing paths: {missing_examples}")
            raise FileNotFoundError(
                f"Critical Error: More than 50% of files are missing in {name} dataset."
            )

    verify_files(meta_train, "Train")
    verify_files(meta_val, "Val")
    verify_files(meta_test, "Test")

    # 2. Verify Split Logic
    # Assertion 1: No overlap between Train and Val images
    train_imgs = set(meta_train["Image"])
    val_imgs = set(meta_val["Image"])
    overlap = train_imgs.intersection(val_imgs)
    assert (
        len(overlap) == 0
    ), f"Found {len(overlap)} overlapping images between Train and Val."

    # Assertion 2: Validation classes must be a subset of Training classes
    # (Because we forced singletons into Train, Val cannot have unique classes)
    train_classes = set(meta_train["Id"].unique())
    val_classes = set(meta_val["Id"].unique())
    unknown_val_classes = val_classes - train_classes
    assert (
        len(unknown_val_classes) == 0
    ), f"Validation set contains {len(unknown_val_classes)} classes not in Training set."

    # Assertion 3: Verify Stratification Ratio on a major class
    # We check 'new_whale' or the most frequent class to ensure the split ratio is approximately correct
    # for classes that were actually split.
    most_frequent_id = meta_train["Id"].mode()[0]
    train_count = len(meta_train[meta_train["Id"] == most_frequent_id])
    val_count = len(meta_val[meta_val["Id"] == most_frequent_id])
    total_count = train_count + val_count

    if total_count > 0:
        actual_val_ratio = val_count / total_count
        print(
            f"Split ratio for class '{most_frequent_id}': {actual_val_ratio:.4f} (Target: {VAL_SIZE})"
        )

        # Allow small deviation
        assert (
            abs(actual_val_ratio - VAL_SIZE) < 0.05
        ), f"Stratification failed: Class '{most_frequent_id}' has ratio {actual_val_ratio:.4f}, expected ~{VAL_SIZE}"

    print("\nAll verification checks passed.")


if __name__ == "__main__":
    main()
