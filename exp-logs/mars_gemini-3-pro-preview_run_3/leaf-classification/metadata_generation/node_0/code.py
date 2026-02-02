import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def main():
    # ==========================================
    # Configuration & Setup
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    IMAGES_DIR_NAME = "images"
    RANDOM_STATE = 42
    VAL_SIZE = 0.2

    # Create metadata directory
    os.makedirs(METADATA_DIR, exist_ok=True)

    # ==========================================
    # Data Loading & Processing
    # ==========================================
    print("Loading data from input directory...")
    train_csv_path = os.path.join(INPUT_DIR, "train.csv")
    test_csv_path = os.path.join(INPUT_DIR, "test.csv")

    # Read CSVs
    df_train_full = pd.read_csv(train_csv_path)
    df_test = pd.read_csv(test_csv_path)

    # Generate relative file paths
    # The images are stored as {id}.jpg in the images/ subdirectory
    def get_rel_path(row):
        return os.path.join(IMAGES_DIR_NAME, f"{int(row['id'])}.jpg")

    df_train_full["file_path"] = df_train_full.apply(get_rel_path, axis=1)
    df_test["file_path"] = df_test.apply(get_rel_path, axis=1)

    # ==========================================
    # Splitting (Train/Val)
    # ==========================================
    print("Splitting training data...")
    if "species" not in df_train_full.columns:
        raise ValueError("Column 'species' missing from training data.")

    y = df_train_full["species"]

    # Stratified split
    df_train, df_val = train_test_split(
        df_train_full,
        test_size=VAL_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    # ==========================================
    # Saving Metadata
    # ==========================================
    print("Saving metadata...")
    train_meta_path = os.path.join(METADATA_DIR, "train.csv")
    val_meta_path = os.path.join(METADATA_DIR, "val.csv")
    test_meta_path = os.path.join(METADATA_DIR, "test.csv")

    df_train.to_csv(train_meta_path, index=False)
    df_val.to_csv(val_meta_path, index=False)
    df_test.to_csv(test_meta_path, index=False)

    # ==========================================
    # Verification & Quality Checks
    # ==========================================
    print("\nPerforming verification checks...")

    # Reload data to ensure integrity
    meta_train = pd.read_csv(train_meta_path)
    meta_val = pd.read_csv(val_meta_path)
    meta_test = pd.read_csv(test_meta_path)

    # 1. Summary Statistics
    print("\n--- Summary Statistics ---")
    print(f"Train samples: {len(meta_train)}")
    print(f"Val samples:   {len(meta_val)}")
    print(f"Test samples:  {len(meta_test)}")

    print("\nTrain Class Distribution (Top 3):")
    print(meta_train["species"].value_counts().head(3))

    # 2. File Path Verification
    def check_files(df, name):
        # Sample up to 1000 paths
        sample_size = min(1000, len(df))
        sample = df.sample(n=sample_size, random_state=RANDOM_STATE)
        paths = sample["file_path"].tolist()

        missing_count = 0
        missing_samples = []

        for p in paths:
            full_path = os.path.join(INPUT_DIR, p)
            if not os.path.exists(full_path):
                missing_count += 1
                if len(missing_samples) < 5:
                    missing_samples.append(p)

        ratio = missing_count / sample_size
        print(f"\nMissing file ratio for {name}: {ratio:.4f}")

        if ratio > 0.5:
            print(f"Sample missing paths: {missing_samples}")
            raise FileNotFoundError(
                f"More than 50% of files are missing in {name} dataset."
            )

    check_files(meta_train, "Train")
    check_files(meta_val, "Validation")
    check_files(meta_test, "Test")

    # 3. Stratification Verification
    print("\nVerifying stratification...")
    train_dist = meta_train["species"].value_counts(normalize=True)
    val_dist = meta_val["species"].value_counts(normalize=True)

    # Check if all validation classes exist in training
    val_classes = set(val_dist.index)
    train_classes = set(train_dist.index)

    if not val_classes.issubset(train_classes):
        diff = val_classes - train_classes
        raise AssertionError(
            f"Validation set contains classes not in training set: {diff}"
        )

    # Check distribution consistency
    # We calculate the maximum absolute difference in class probabilities
    all_classes = sorted(list(train_classes | val_classes))
    diffs = []
    for c in all_classes:
        p_train = train_dist.get(c, 0)
        p_val = val_dist.get(c, 0)
        diffs.append(abs(p_train - p_val))

    max_diff = max(diffs)
    print(f"Max difference in class proportions: {max_diff:.6f}")

    # Tolerance: With small datasets, exact stratification is impossible.
    # A tolerance of 0.05 allows for discrete quantization errors in small classes.
    if max_diff > 0.05:
        raise AssertionError(
            f"Stratification failed. Max class proportion difference {max_diff} > 0.05"
        )

    print("\nAll checks passed successfully.")


if __name__ == "__main__":
    main()
