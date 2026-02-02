import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split

# Configuration
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
TRAIN_META_FILE = os.path.join(INPUT_DIR, "train_metadata.json")
TEST_META_FILE = os.path.join(INPUT_DIR, "test_metadata.json")
RANDOM_STATE = 42


def resolve_path_strategy(sample_fname, root_dir, check_dir):
    """
    Determines the correct relative path construction by checking a sample file against the filesystem.

    Args:
        sample_fname: The filename found in the metadata JSON.
        root_dir: The base input directory (e.g., './input').
        check_dir: The expected subdirectory (e.g., 'train_images' or 'test_images').

    Returns:
        A function that takes a filename and returns the correct relative path.
    """
    # Strategy 1: Prepend the check_dir (e.g., 'train_images/' + '000/00/img.jpg')
    p1 = os.path.join(check_dir, sample_fname)
    if os.path.exists(os.path.join(root_dir, p1)):
        return lambda x: os.path.join(check_dir, x)

    # Strategy 2: Use as is (e.g., if fname already includes 'train_images/')
    p2 = sample_fname
    if os.path.exists(os.path.join(root_dir, p2)):
        return lambda x: x

    # Strategy 3: Strip known prefixes (e.g., 'h22-train/images/') and prepend check_dir
    # This handles cases where metadata has a different root prefix than the actual folder.
    prefixes = ["h22-train/images/", "images/"]
    for prefix in prefixes:
        if sample_fname.startswith(prefix):
            clean_name = sample_fname[len(prefix) :]
            p3 = os.path.join(check_dir, clean_name)
            if os.path.exists(os.path.join(root_dir, p3)):
                return lambda x: os.path.join(check_dir, x[len(prefix) :])

    # Fallback: Default to Strategy 1 (will likely trigger validation errors if wrong)
    return lambda x: os.path.join(check_dir, x)


def validate_dataset(df, name):
    """
    Validates the dataset by checking file existence and printing stats.
    """
    print(f"\n--- {name} Set ---")
    print(f"Total Samples: {len(df)}")
    if "category_id" in df.columns:
        print(f"Unique Classes: {df['category_id'].nunique()}")
        print(f"Class Distribution:\n{df['category_id'].value_counts().describe()}")

    if len(df) == 0:
        return

    # Check for missing files
    sample_n = min(1000, len(df))
    sample_df = df.sample(n=sample_n, random_state=RANDOM_STATE)

    missing_count = 0
    missing_examples = []

    for _, row in sample_df.iterrows():
        full_path = os.path.join(INPUT_DIR, row["file_path"])
        if not os.path.exists(full_path):
            missing_count += 1
            if len(missing_examples) < 5:
                missing_examples.append(row["file_path"])

    ratio = missing_count / sample_n
    print(f"Missing File Ratio: {ratio:.4f}")

    if ratio > 0.5:
        print(f"Sample missing files: {missing_examples}")
        raise FileNotFoundError(
            f"Validation Failed: More than 50% of files are missing in {name} dataset."
        )


def main():
    os.makedirs(METADATA_DIR, exist_ok=True)

    # ---------------------------------------------------------
    # 1. Process Training Metadata
    # ---------------------------------------------------------
    print("Loading train_metadata.json...")
    with open(TRAIN_META_FILE, "r") as f:
        train_meta = json.load(f)

    # train_metadata.json is expected to be a dict with 'images' and 'annotations'
    if not isinstance(train_meta, dict):
        raise ValueError(
            "Unexpected format for train_metadata.json (expected dictionary)."
        )

    images_df = pd.DataFrame(train_meta["images"])
    annotations_df = pd.DataFrame(train_meta["annotations"])

    # Ensure image_id is string for consistent merging
    images_df["image_id"] = images_df["image_id"].astype(str)
    annotations_df["image_id"] = annotations_df["image_id"].astype(str)

    print("Merging training images and annotations...")
    train_df = pd.merge(images_df, annotations_df, on="image_id", how="inner")

    # Determine path strategy for training images
    if not train_df.empty:
        sample_fname = train_df["file_name"].iloc[0]
        path_func = resolve_path_strategy(sample_fname, INPUT_DIR, "train_images")
        train_df["file_path"] = train_df["file_name"].apply(path_func)

    # ---------------------------------------------------------
    # 2. Split Training Data (Train/Val)
    # ---------------------------------------------------------
    print("Splitting training data...")
    # Separate classes with < 2 samples (cannot be stratified)
    counts = train_df["category_id"].value_counts()
    singletons = counts[counts < 2].index

    singleton_mask = train_df["category_id"].isin(singletons)
    df_singletons = train_df[singleton_mask].copy()
    df_rest = train_df[~singleton_mask].copy()

    if len(df_rest) > 0:
        # Stratified split for the rest
        train_split, val_split = train_test_split(
            df_rest,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=df_rest["category_id"],
        )
        # Add singletons back to train to preserve all classes
        final_train = pd.concat([df_singletons, train_split])
        final_val = val_split
    else:
        final_train = df_singletons
        final_val = pd.DataFrame(columns=train_df.columns)

    # ---------------------------------------------------------
    # 3. Process Test Metadata
    # ---------------------------------------------------------
    print("Loading test_metadata.json...")
    with open(TEST_META_FILE, "r") as f:
        test_meta = json.load(f)

    # test_metadata.json is expected to be a list of records
    if isinstance(test_meta, list):
        test_df = pd.DataFrame(test_meta)
    else:
        raise ValueError("Unexpected format for test_metadata.json (expected list).")

    # Determine path strategy for test images
    if not test_df.empty:
        sample_fname_test = test_df["file_name"].iloc[0]
        path_func_test = resolve_path_strategy(
            sample_fname_test, INPUT_DIR, "test_images"
        )
        test_df["file_path"] = test_df["file_name"].apply(path_func_test)
    else:
        test_df["file_path"] = []

    # ---------------------------------------------------------
    # 4. Save Metadata
    # ---------------------------------------------------------
    print("Saving metadata CSVs...")
    cols_train = ["image_id", "file_path", "category_id"]
    cols_test = ["image_id", "file_path"]

    final_train[cols_train].to_csv(os.path.join(METADATA_DIR, "train.csv"), index=False)
    final_val[cols_train].to_csv(os.path.join(METADATA_DIR, "val.csv"), index=False)
    test_df[cols_test].to_csv(os.path.join(METADATA_DIR, "test.csv"), index=False)

    # ---------------------------------------------------------
    # 5. Verification
    # ---------------------------------------------------------
    print("Verifying datasets...")
    validate_dataset(final_train, "Train")
    validate_dataset(final_val, "Validation")
    validate_dataset(test_df, "Test")

    # Verify Stratification Logic
    if not final_val.empty:
        train_classes = set(final_train["category_id"].unique())
        val_classes = set(final_val["category_id"].unique())

        # Validation classes must be a subset of training classes
        if not val_classes.issubset(train_classes):
            diff = val_classes - train_classes
            raise AssertionError(
                f"Validation set contains {len(diff)} classes not present in Training set!"
            )
        print(
            "Stratification check passed: All validation classes are present in training set."
        )

    print("\nMetadata generation and validation completed successfully.")


if __name__ == "__main__":
    main()
