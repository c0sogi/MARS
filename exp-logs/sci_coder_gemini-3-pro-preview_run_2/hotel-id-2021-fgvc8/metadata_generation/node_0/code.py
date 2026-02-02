import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split

# Constants
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
RANDOM_STATE = 42
VAL_SIZE = 0.2


def generate_metadata():
    """
    Reads raw data, generates file paths, performs stratified split,
    and saves metadata CSVs.
    """
    # Ensure metadata directory exists
    os.makedirs(METADATA_DIR, exist_ok=True)

    print("Loading raw data...")
    train_csv_path = os.path.join(INPUT_DIR, "train.csv")
    sample_sub_path = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Read CSVs
    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(sample_sub_path)

    # Construct relative file paths
    # Train structure: train_images/<chain_id>/<image_id>
    # Note: chain is an integer in csv, but folder name is string.
    train_df["file_path"] = (
        "train_images/" + train_df["chain"].astype(str) + "/" + train_df["image"]
    )

    # Test structure: test_images/<image_id>
    test_df["file_path"] = "test_images/" + test_df["image"]

    print("Splitting data...")
    # Analyze class distribution
    hotel_counts = train_df["hotel_id"].value_counts()

    # Identify singletons (hotels with < 2 images)
    # These cannot be stratified split, so we assign them to train to ensure model sees them.
    singletons = hotel_counts[hotel_counts < 2].index
    multi_samples = hotel_counts[hotel_counts >= 2].index

    df_singletons = train_df[train_df["hotel_id"].isin(singletons)].copy()
    df_multi = train_df[train_df["hotel_id"].isin(multi_samples)].copy()

    print(f"  Total training images: {len(train_df)}")
    print(
        f"  Classes with < 2 images (forced to train): {len(singletons)} classes, {len(df_singletons)} images"
    )
    print(
        f"  Classes with >= 2 images (stratified split): {len(multi_samples)} classes, {len(df_multi)} images"
    )

    # Perform stratified split on multi-sample data
    train_split, val_split = train_test_split(
        df_multi,
        test_size=VAL_SIZE,
        stratify=df_multi["hotel_id"],
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    # Combine singletons with the training split
    train_final = pd.concat([train_split, df_singletons], axis=0)

    # Shuffle the final training set
    train_final = train_final.sample(frac=1, random_state=RANDOM_STATE).reset_index(
        drop=True
    )
    val_final = val_split.reset_index(drop=True)

    # Save metadata
    print("Saving metadata files...")
    train_final.to_csv(os.path.join(METADATA_DIR, "train_metadata.csv"), index=False)
    val_final.to_csv(os.path.join(METADATA_DIR, "val_metadata.csv"), index=False)
    test_df.to_csv(os.path.join(METADATA_DIR, "test_metadata.csv"), index=False)

    return train_final, val_final, test_df


def validate_metadata(train_df, val_df, test_df):
    """
    Performs validation checks on the generated metadata.
    """
    print("\n=== Validating Metadata ===")

    # 1. Summary Statistics
    print("\n[Summary Statistics]")
    print(
        f"Train Set: {len(train_df)} images, {train_df['hotel_id'].nunique()} unique hotels"
    )
    print(
        f"Val Set:   {len(val_df)} images, {val_df['hotel_id'].nunique()} unique hotels"
    )
    print(f"Test Set:  {len(test_df)} images")

    # 2. Check File Existence
    print("\n[File Existence Check]")

    def check_files(df, name):
        if df.empty:
            print(f"{name} is empty, skipping file check.")
            return

        # Sample 1000 paths (or all if less than 1000)
        sample_size = min(1000, len(df))
        sample = df.sample(n=sample_size, random_state=RANDOM_STATE)

        missing_count = 0
        missing_examples = []

        for _, row in sample.iterrows():
            rel_path = row["file_path"]
            # Full path is ./input/ + relative_path
            full_path = os.path.join(INPUT_DIR, rel_path)

            if not os.path.exists(full_path):
                missing_count += 1
                if len(missing_examples) < 5:
                    missing_examples.append(rel_path)

        ratio = missing_count / sample_size
        print(f"{name}: Checked {sample_size} files. Missing ratio: {ratio:.4f}")

        if ratio > 0.5:
            print(f"Example missing files in {name}:")
            for p in missing_examples:
                print(f"  - {p}")
            raise FileNotFoundError(
                f"More than 50% of files missing in {name} dataset."
            )

    check_files(train_df, "Train")
    check_files(val_df, "Validation")
    check_files(test_df, "Test")

    # 3. Verify Split Requirements
    print("\n[Split Verification]")

    # Check for data leakage (intersection of images)
    train_imgs = set(train_df["image"])
    val_imgs = set(val_df["image"])
    intersection = train_imgs.intersection(val_imgs)

    if intersection:
        raise AssertionError(
            f"Data leakage detected! {len(intersection)} images are in both Train and Validation sets."
        )
    else:
        print("Success: No image overlap between Train and Validation.")

    # Check that all validation hotels exist in training
    # (Since we put singletons in train, train should contain a superset of val classes)
    train_hotels = set(train_df["hotel_id"])
    val_hotels = set(val_df["hotel_id"])

    missing_hotels = val_hotels - train_hotels
    if missing_hotels:
        raise AssertionError(
            f"Validation set contains {len(missing_hotels)} hotels not present in Training set."
        )
    else:
        print("Success: All Validation hotels are present in Training set.")

    print("\nAll validation checks passed.")


if __name__ == "__main__":
    try:
        train_meta, val_meta, test_meta = generate_metadata()
        validate_metadata(train_meta, val_meta, test_meta)
    except Exception as e:
        print(f"\nERROR: Script failed with exception: {e}")
        raise e
