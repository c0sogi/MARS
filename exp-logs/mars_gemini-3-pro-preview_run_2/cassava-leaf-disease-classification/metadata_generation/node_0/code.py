import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def main():
    # Configuration
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    RANDOM_STATE = 42
    VAL_SIZE = 0.2

    # Create metadata directory
    os.makedirs(METADATA_DIR, exist_ok=True)

    print("Loading raw data...")
    # Load training data
    train_csv_path = os.path.join(INPUT_DIR, "train.csv")
    df_full_train = pd.read_csv(train_csv_path)

    # Load test data (from sample_submission.csv)
    test_csv_path = os.path.join(INPUT_DIR, "sample_submission.csv")
    df_test = pd.read_csv(test_csv_path)

    # Add relative file paths
    # Note: Train images are in 'train_images/' and test images in 'test_images/'
    df_full_train["file_path"] = "train_images/" + df_full_train["image_id"]
    df_test["file_path"] = "test_images/" + df_test["image_id"]

    # Remove label column from test metadata if it exists (it's usually a dummy in sample_submission)
    # We keep it if it's required by the consumer, but usually test metadata just needs IDs/paths.
    # However, for consistency with the prompt "sample_submission.csv... label the predicted ID",
    # we will leave the structure as is but ensure we treat it as test data.

    print(f"Total training samples: {len(df_full_train)}")
    print(f"Total test samples: {len(df_test)}")

    # Perform Stratified Split
    print("Splitting training data into train and validation sets...")
    df_train, df_val = train_test_split(
        df_full_train,
        test_size=VAL_SIZE,
        stratify=df_full_train["label"],
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    # Save metadata
    print("Saving metadata to ./metadata/ ...")
    train_meta_path = os.path.join(METADATA_DIR, "train.csv")
    val_meta_path = os.path.join(METADATA_DIR, "val.csv")
    test_meta_path = os.path.join(METADATA_DIR, "test.csv")

    df_train.to_csv(train_meta_path, index=False)
    df_val.to_csv(val_meta_path, index=False)
    df_test.to_csv(test_meta_path, index=False)

    print("Metadata generation complete.")

    # ==========================================
    # Verification Step
    # ==========================================
    print("\nStarting verification checks...")

    # 1. Load datasets back
    df_train_loaded = pd.read_csv(train_meta_path)
    df_val_loaded = pd.read_csv(val_meta_path)
    df_test_loaded = pd.read_csv(test_meta_path)

    # 2. Print Summary Statistics
    print("\n--- Summary Statistics ---")
    print(f"Train set shape: {df_train_loaded.shape}")
    print(f"Validation set shape: {df_val_loaded.shape}")
    print(f"Test set shape: {df_test_loaded.shape}")

    print("\nTrain Class Distribution:")
    print(df_train_loaded["label"].value_counts(normalize=True))
    print("\nValidation Class Distribution:")
    print(df_val_loaded["label"].value_counts(normalize=True))

    # 3. Check file paths
    def check_filepaths(df, name):
        print(f"\nChecking file paths for {name}...")
        # Sample 1000 or all if less
        n_sample = min(1000, len(df))
        if n_sample == 0:
            print(f"No samples in {name} to check.")
            return

        sample_paths = df["file_path"].sample(n=n_sample, random_state=RANDOM_STATE)

        missing_count = 0
        missing_samples = []

        for rel_path in sample_paths:
            full_path = os.path.join(INPUT_DIR, rel_path)
            if not os.path.exists(full_path):
                missing_count += 1
                if len(missing_samples) < 5:
                    missing_samples.append(rel_path)

        missing_ratio = missing_count / n_sample
        print(f"Missing file ratio: {missing_ratio:.4f} ({missing_count}/{n_sample})")

        if missing_ratio > 0.5:
            print("Sample missing paths:", missing_samples)
            raise FileNotFoundError(
                f"More than 50% of file paths in {name} are missing."
            )

    check_filepaths(df_train_loaded, "Train")
    check_filepaths(df_val_loaded, "Validation")
    check_filepaths(df_test_loaded, "Test")

    # 4. Verify Validation Split Requirements
    print("\nVerifying split logic...")

    # Check 1: No overlap
    train_ids = set(df_train_loaded["image_id"])
    val_ids = set(df_val_loaded["image_id"])
    intersection = train_ids.intersection(val_ids)
    assert (
        len(intersection) == 0
    ), f"Train and Validation sets overlap! {len(intersection)} common IDs."

    # Check 2: Ratio
    total_train_val = len(df_train_loaded) + len(df_val_loaded)
    val_ratio = len(df_val_loaded) / total_train_val
    print(f"Actual Validation Ratio: {val_ratio:.4f}")
    assert 0.19 < val_ratio < 0.21, "Validation ratio is not approximately 0.2"

    # Check 3: Stratification
    # We check if the distribution of labels in train and val are roughly similar (within a tolerance)
    train_dist = df_train_loaded["label"].value_counts(normalize=True).sort_index()
    val_dist = df_val_loaded["label"].value_counts(normalize=True).sort_index()

    # Calculate max absolute difference in proportions
    diffs = (train_dist - val_dist).abs()
    max_diff = diffs.max()
    print(f"Max difference in class proportions between Train and Val: {max_diff:.4f}")

    # Tolerance of 0.02 (2%) is usually sufficient for stratified splits on reasonable dataset sizes
    assert (
        max_diff < 0.02
    ), "Stratification failed: Class distributions differ significantly."

    print("\nAll checks passed successfully.")


if __name__ == "__main__":
    main()
