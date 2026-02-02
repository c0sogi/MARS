import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Configuration
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
RANDOM_STATE = 42


def generate_metadata():
    """
    Generates metadata CSVs for train, val, and test sets.
    """
    # Create metadata directory
    if not os.path.exists(METADATA_DIR):
        os.makedirs(METADATA_DIR)

    # --- Process Training Data ---
    print("Processing training data...")
    train_csv_path = os.path.join(INPUT_DIR, "train.csv")
    if not os.path.exists(train_csv_path):
        raise FileNotFoundError(f"{train_csv_path} not found.")

    train_df = pd.read_csv(train_csv_path)

    # Remove duplicate images to prevent data leakage
    train_df = train_df.drop_duplicates(subset=["image"], keep="first").reset_index(
        drop=True
    )

    # Construct relative file paths
    # Structure: train_images/<chain_id>/<image_id>
    # Ensure chain is integer for directory name matching
    train_df["chain"] = train_df["chain"].astype(int)
    train_df["file_path"] = (
        "train_images/" + train_df["chain"].astype(str) + "/" + train_df["image"]
    )

    # Split into Train and Validation
    # Strategy:
    # 1. Identify classes with < 2 samples (singletons). These cannot be stratified.
    # 2. Put singletons in Train to ensure the model sees them.
    # 3. Stratify split the remaining multi-sample classes 80:20.

    hotel_counts = train_df["hotel_id"].value_counts()
    singletons = hotel_counts[hotel_counts < 2].index
    multi_samples = hotel_counts[hotel_counts >= 2].index

    print(f"Total classes: {len(hotel_counts)}")
    print(f"Classes with < 2 samples: {len(singletons)}")
    print(f"Classes with >= 2 samples: {len(multi_samples)}")

    df_single = train_df[train_df["hotel_id"].isin(singletons)].copy()
    df_multi = train_df[train_df["hotel_id"].isin(multi_samples)].copy()

    # Stratified split for multi-sample classes
    train_multi, val_multi = train_test_split(
        df_multi,
        test_size=0.2,
        stratify=df_multi["hotel_id"],
        random_state=RANDOM_STATE,
    )

    # Combine singletons into train
    final_train = pd.concat([df_single, train_multi], ignore_index=True)
    final_val = val_multi.copy()

    # Shuffle final datasets
    final_train = final_train.sample(frac=1, random_state=RANDOM_STATE).reset_index(
        drop=True
    )
    final_val = final_val.sample(frac=1, random_state=RANDOM_STATE).reset_index(
        drop=True
    )

    # Save to metadata
    train_meta_path = os.path.join(METADATA_DIR, "train.csv")
    val_meta_path = os.path.join(METADATA_DIR, "val.csv")

    final_train.to_csv(train_meta_path, index=False)
    final_val.to_csv(val_meta_path, index=False)

    # --- Process Test Data ---
    print("Processing test data...")
    # We use sample_submission.csv to define the test set images
    sample_sub_path = os.path.join(INPUT_DIR, "sample_submission.csv")
    if not os.path.exists(sample_sub_path):
        raise FileNotFoundError(f"{sample_sub_path} not found.")

    test_df = pd.read_csv(sample_sub_path)

    # Construct relative file paths
    # Structure: test_images/<image_id>
    test_df["file_path"] = "test_images/" + test_df["image"]

    test_meta_path = os.path.join(METADATA_DIR, "test.csv")
    test_df.to_csv(test_meta_path, index=False)

    return final_train, final_val, test_df


def validate_datasets(train_df, val_df, test_df):
    """
    Validates the generated datasets against requirements.
    """
    print("\nValidating datasets...")

    # 1. Summary Statistics
    print("-" * 30)
    print(f"Train Set: {len(train_df)} samples")
    print(f"Val Set:   {len(val_df)} samples")
    print(f"Test Set:  {len(test_df)} samples")
    print(f"Train unique hotels: {train_df['hotel_id'].nunique()}")
    print(f"Val unique hotels:   {val_df['hotel_id'].nunique()}")
    print("-" * 30)

    # 2. Check File Existence (Random Sample)
    def check_paths(df, name):
        if "file_path" not in df.columns:
            raise ValueError(f"{name} dataframe missing 'file_path' column")

        sample_n = min(1000, len(df))
        sample = df.sample(n=sample_n, random_state=RANDOM_STATE)

        missing = 0
        missing_examples = []
        for _, row in sample.iterrows():
            # Path is relative to INPUT_DIR
            full_path = os.path.join(INPUT_DIR, row["file_path"])
            if not os.path.exists(full_path):
                missing += 1
                if len(missing_examples) < 5:
                    missing_examples.append(row["file_path"])

        ratio = missing / sample_n
        print(f"{name}: Missing file ratio = {ratio:.4f} ({missing}/{sample_n})")

        if ratio > 0.5:
            print(f"Examples of missing files in {name}:")
            for p in missing_examples:
                print(f"  {p}")
            raise FileNotFoundError(
                f"Too many missing files in {name} dataset (ratio {ratio:.2f} > 0.5)"
            )

    check_paths(train_df, "Train")
    check_paths(val_df, "Val")
    check_paths(test_df, "Test")

    # 3. Verify Validation Split
    # Check for leakage
    train_imgs = set(train_df["image"])
    val_imgs = set(val_df["image"])
    intersection = train_imgs.intersection(val_imgs)
    if intersection:
        raise AssertionError(
            f"Data leakage detected! {len(intersection)} images are in both train and val."
        )

    # Check stratification logic
    # We expect val distribution to roughly match train distribution for multi-sample classes
    # Just check that val classes are a subset of train classes (since we put singletons in train)
    train_classes = set(train_df["hotel_id"])
    val_classes = set(val_df["hotel_id"])

    if not val_classes.issubset(train_classes):
        diff = val_classes - train_classes
        raise AssertionError(
            f"Validation set contains classes not in training set: {diff}"
        )

    print("Validation split verification passed.")


if __name__ == "__main__":
    try:
        train_df, val_df, test_df = generate_metadata()
        validate_datasets(train_df, val_df, test_df)
        print("\nMetadata generation and validation successful.")
    except Exception as e:
        print(f"\nERROR: {e}")
        raise e
