import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def main():
    # Define directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    IMAGES_DIR = "images"  # Relative to input

    # Create metadata directory
    os.makedirs(METADATA_DIR, exist_ok=True)

    # Load raw data
    train_csv_path = os.path.join(INPUT_DIR, "train.csv")
    test_csv_path = os.path.join(INPUT_DIR, "test.csv")

    print(f"Loading data from {INPUT_DIR}...")
    df_train_full = pd.read_csv(train_csv_path)
    df_test = pd.read_csv(test_csv_path)

    # Identify target columns
    # Based on description and sample submission, targets are healthy, rust, scab, and either multiple_diseases or combinations
    # We will detect them dynamically excluding image_id
    potential_targets = [col for col in df_train_full.columns if col != "image_id"]
    print(f"Identified target columns: {potential_targets}")

    # Construct relative file paths
    # Image IDs in CSV do not have extensions, files in images/ have .jpg extension
    def get_rel_path(image_id):
        return os.path.join(IMAGES_DIR, f"{image_id}.jpg")

    df_train_full["file_path"] = df_train_full["image_id"].apply(get_rel_path)
    df_test["file_path"] = df_test["image_id"].apply(get_rel_path)

    # Prepare for stratified split
    # We need to create a single label for stratification.
    # Since this is a multi-class problem (one-hot encoded in the provided CSV usually),
    # we can use argmax to get the class index/name.
    # We assume the rows are one-hot encoded or represent a probability distribution where one class dominates.
    # Given the description "distinguish between leaves...", it's likely single-label per image in terms of ground truth categories (even if 'multiple_diseases' is a category).

    # Create a 'stratify_label' column
    df_train_full["stratify_label"] = df_train_full[potential_targets].idxmax(axis=1)

    # Perform Stratified Split (80/20)
    print("Splitting training data into train and validation sets (80:20)...")
    df_train, df_val = train_test_split(
        df_train_full,
        test_size=0.20,
        stratify=df_train_full["stratify_label"],
        random_state=42,
        shuffle=True,
    )

    # Save metadata
    train_meta_path = os.path.join(METADATA_DIR, "train_metadata.csv")
    val_meta_path = os.path.join(METADATA_DIR, "val_metadata.csv")
    test_meta_path = os.path.join(METADATA_DIR, "test_metadata.csv")

    # We keep the stratify_label for verification but it's not strictly required for the final metadata if not needed by model
    # However, keeping all original columns + file_path is good practice.

    df_train.to_csv(train_meta_path, index=False)
    df_val.to_csv(val_meta_path, index=False)
    df_test.to_csv(test_meta_path, index=False)

    print(f"Metadata saved to {METADATA_DIR}")

    # ==========================================
    # Verification Steps
    # ==========================================
    print("\nStarting verification...")

    # 1. Load the generated metadata
    df_train_loaded = pd.read_csv(train_meta_path)
    df_val_loaded = pd.read_csv(val_meta_path)
    df_test_loaded = pd.read_csv(test_meta_path)

    # 2. Print Summary Statistics
    print("\nSummary Statistics:")
    print(f"Train set shape: {df_train_loaded.shape}")
    print(f"Validation set shape: {df_val_loaded.shape}")
    print(f"Test set shape: {df_test_loaded.shape}")

    print("\nClass Distribution in Train:")
    print(df_train_loaded["stratify_label"].value_counts(normalize=True))
    print("\nClass Distribution in Validation:")
    print(df_val_loaded["stratify_label"].value_counts(normalize=True))

    # 3. Check file paths
    def check_files(df, name):
        print(f"\nChecking file paths for {name} dataset...")
        # Sample up to 1000 paths
        sample_size = min(1000, len(df))
        sample_paths = df["file_path"].sample(n=sample_size, random_state=42).tolist()

        missing_count = 0
        missing_samples = []

        for rel_path in sample_paths:
            full_path = os.path.join(INPUT_DIR, rel_path)
            if not os.path.exists(full_path):
                missing_count += 1
                if len(missing_samples) < 5:
                    missing_samples.append(rel_path)

        missing_ratio = missing_count / sample_size
        print(
            f"Missing file ratio: {missing_ratio:.4f} ({missing_count}/{sample_size})"
        )

        if missing_ratio > 0.5:
            print("Sample missing files:")
            for p in missing_samples:
                print(f"  {p}")
            raise FileNotFoundError(
                f"More than 50% of file paths in {name} metadata do not exist."
            )

    check_files(df_train_loaded, "Train")
    check_files(df_val_loaded, "Validation")
    check_files(df_test_loaded, "Test")

    # 4. Verify Stratification
    print("\nVerifying stratification...")
    train_dist = df_train_loaded["stratify_label"].value_counts(normalize=True)
    val_dist = df_val_loaded["stratify_label"].value_counts(normalize=True)

    # Check if distributions are similar (within a tolerance)
    # Since datasets might be small, exact match isn't expected, but should be close.
    # We'll check if the absolute difference in proportions is small (< 0.05 for each class)
    for label in train_dist.index:
        train_prop = train_dist.get(label, 0)
        val_prop = val_dist.get(label, 0)
        diff = abs(train_prop - val_prop)
        print(
            f"Class '{label}': Train={train_prop:.4f}, Val={val_prop:.4f}, Diff={diff:.4f}"
        )

        if diff > 0.05:
            raise AssertionError(
                f"Stratification failed for class {label}. Difference {diff:.4f} > 0.05"
            )

    print("\nAll checks passed successfully.")


if __name__ == "__main__":
    main()
