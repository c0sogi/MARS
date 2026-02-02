import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split


def main():
    # Constants
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    TRAIN_DIR = os.path.join(INPUT_DIR, "train")
    TEST_DIR = os.path.join(INPUT_DIR, "test")
    RANDOM_STATE = 42

    # Ensure metadata directory exists
    os.makedirs(METADATA_DIR, exist_ok=True)

    print("Loading datasets...")
    train_csv_path = os.path.join(INPUT_DIR, "train.csv")
    sample_sub_path = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Read CSVs
    df_train_orig = pd.read_csv(train_csv_path)
    df_sample_sub = pd.read_csv(sample_sub_path)

    # --- Process Training Data ---
    # Generate file paths
    # Assuming .dicom extension based on dataset description
    df_train_orig["file_path"] = df_train_orig["image_id"].apply(
        lambda x: os.path.join(TRAIN_DIR, f"{x}.dicom")
    )

    # Group Stratified Split
    # 1. Get unique images
    unique_images = df_train_orig["image_id"].unique()

    # 2. Create stratification labels per image
    # We stratify based on "Has Finding" vs "No Finding" (Class 14)
    # Group by image_id and check if class_id 14 is the only class present
    # Note: class_id 14 is "No finding".
    # If an image has class 14, it usually implies no other findings.
    # If an image has other classes, it is a "Finding" image.

    print("Performing Group Stratified Split...")
    image_labels = df_train_orig.groupby("image_id")["class_id"].apply(set)

    stratify_labels = []
    for img_id in unique_images:
        labels = image_labels[img_id]
        # If the set of labels contains 14 and is length 1, it's a "No finding" image.
        # Otherwise, it contains findings.
        if 14 in labels and len(labels) == 1:
            stratify_labels.append(0)  # Label 0: No Finding
        else:
            stratify_labels.append(1)  # Label 1: Has Finding

    # 3. Split unique images
    train_imgs, val_imgs = train_test_split(
        unique_images,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=stratify_labels,
    )

    # 4. Filter original dataframe to create train/val sets
    train_meta = df_train_orig[df_train_orig["image_id"].isin(train_imgs)].copy()
    val_meta = df_train_orig[df_train_orig["image_id"].isin(val_imgs)].copy()

    # --- Process Test Data ---
    # The sample submission contains the test image_ids
    test_meta = df_sample_sub[["image_id"]].copy()
    test_meta["file_path"] = test_meta["image_id"].apply(
        lambda x: os.path.join(TEST_DIR, f"{x}.dicom")
    )

    # --- Save Metadata ---
    print("Saving metadata...")
    train_meta.to_csv(os.path.join(METADATA_DIR, "train_meta.csv"), index=False)
    val_meta.to_csv(os.path.join(METADATA_DIR, "val_meta.csv"), index=False)
    test_meta.to_csv(os.path.join(METADATA_DIR, "test_meta.csv"), index=False)

    # --- Validation Checks ---
    print("\n" + "=" * 30)
    print("VALIDATION CHECKS")
    print("=" * 30)

    # Reload data
    df_train_new = pd.read_csv(os.path.join(METADATA_DIR, "train_meta.csv"))
    df_val_new = pd.read_csv(os.path.join(METADATA_DIR, "val_meta.csv"))
    df_test_new = pd.read_csv(os.path.join(METADATA_DIR, "test_meta.csv"))

    # 1. Summary Statistics
    print(
        f"Train set: {len(df_train_new)} rows, {df_train_new['image_id'].nunique()} unique images"
    )
    print(
        f"Val set:   {len(df_val_new)} rows, {df_val_new['image_id'].nunique()} unique images"
    )
    print(
        f"Test set:  {len(df_test_new)} rows, {df_test_new['image_id'].nunique()} unique images"
    )

    # 2. Check Split Ratio
    n_train = df_train_new["image_id"].nunique()
    n_val = df_val_new["image_id"].nunique()
    total = n_train + n_val
    val_ratio = n_val / total
    print(f"Validation Split Ratio (by images): {val_ratio:.4f} (Target: 0.20)")

    # 3. Check Data Leakage (Group Split Verification)
    train_ids = set(df_train_new["image_id"].unique())
    val_ids = set(df_val_new["image_id"].unique())
    intersection = train_ids.intersection(val_ids)
    if intersection:
        raise AssertionError(
            f"Data Leakage Detected! {len(intersection)} images found in both train and val sets."
        )
    print("Check Passed: No data leakage between train and val.")

    # 4. Check Stratification
    def get_no_finding_ratio(df):
        # Calculate ratio of images that are "No finding"
        # Group by image, check if class 14 is the only class
        img_groups = df.groupby("image_id")["class_id"].apply(lambda x: set(x) == {14})
        return img_groups.mean()

    train_nf_ratio = get_no_finding_ratio(df_train_new)
    val_nf_ratio = get_no_finding_ratio(df_val_new)

    print(f"Train 'No Finding' Image Ratio: {train_nf_ratio:.4f}")
    print(f"Val 'No Finding' Image Ratio:   {val_nf_ratio:.4f}")

    if abs(train_nf_ratio - val_nf_ratio) > 0.05:
        print("WARNING: Stratification distribution differs significantly (> 5%).")
    else:
        print("Check Passed: Stratification distribution is consistent.")

    # 5. Check File Paths
    def check_paths(df, name):
        paths = df["file_path"].values
        # Randomly sample 1000 paths if dataset is larger
        if len(paths) > 1000:
            paths = np.random.choice(paths, 1000, replace=False)

        missing_count = 0
        sample_missing = []

        for p in paths:
            if not os.path.exists(p):
                missing_count += 1
                if len(sample_missing) < 5:
                    sample_missing.append(p)

        ratio = missing_count / len(paths)
        print(f"Missing file ratio for {name}: {ratio:.4f}")

        if ratio > 0.5:
            print("Sample missing paths:", sample_missing)
            raise FileNotFoundError(
                f"Error: More than 50% of file paths in {name} metadata are invalid."
            )

    check_paths(df_train_new, "Train")
    check_paths(df_val_new, "Val")
    check_paths(df_test_new, "Test")

    print("\nMetadata generation and validation completed successfully.")


if __name__ == "__main__":
    main()
