import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def main():
    # Define directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Create metadata directory if it doesn't exist
    os.makedirs(METADATA_DIR, exist_ok=True)

    print("Loading raw data...")
    # Load raw csv files
    train_df = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv"))

    # Add relative file paths
    # Based on dataset info: train images in 'train/', test images in 'test/'
    train_df["file_path"] = train_df["id"].apply(lambda x: os.path.join("train", x))
    test_df["file_path"] = test_df["id"].apply(lambda x: os.path.join("test", x))

    # Split training data into train and validation (80:20)
    # Stratified by 'has_cactus'
    print("Splitting data...")
    X = train_df
    y = train_df["has_cactus"]

    train_split, val_split = train_test_split(
        X, test_size=0.2, random_state=42, stratify=y
    )

    # Save metadata
    print("Saving metadata...")
    train_metadata_path = os.path.join(METADATA_DIR, "train_metadata.csv")
    val_metadata_path = os.path.join(METADATA_DIR, "val_metadata.csv")
    test_metadata_path = os.path.join(METADATA_DIR, "test_metadata.csv")

    train_split.to_csv(train_metadata_path, index=False)
    val_split.to_csv(val_metadata_path, index=False)
    test_df.to_csv(test_metadata_path, index=False)

    # --- Verification Steps ---
    print("\n--- Verifying Metadata ---")

    # 1. Load datasets back
    df_train_check = pd.read_csv(train_metadata_path)
    df_val_check = pd.read_csv(val_metadata_path)
    df_test_check = pd.read_csv(test_metadata_path)

    # 2. Print Summary Statistics
    print(f"\nTraining Set Shape: {df_train_check.shape}")
    print(f"Validation Set Shape: {df_val_check.shape}")
    print(f"Test Set Shape: {df_test_check.shape}")

    print("\nClass Distribution (has_cactus):")
    train_dist = df_train_check["has_cactus"].value_counts(normalize=True)
    val_dist = df_val_check["has_cactus"].value_counts(normalize=True)
    print(f"Train:\n{train_dist}")
    print(f"Val:\n{val_dist}")

    # 3. Check File Paths
    def check_files_exist(df, name):
        print(f"\nChecking file existence for {name}...")
        # Sample 1000 or all if less than 1000
        n_sample = min(1000, len(df))
        sample = df.sample(n=n_sample, random_state=42)

        missing_count = 0
        missing_samples = []

        for _, row in sample.iterrows():
            rel_path = row["file_path"]
            full_path = os.path.join(INPUT_DIR, rel_path)
            if not os.path.exists(full_path):
                missing_count += 1
                if len(missing_samples) < 5:
                    missing_samples.append(rel_path)

        missing_ratio = missing_count / n_sample
        print(f"Missing File Ratio: {missing_ratio:.4f}")

        if missing_ratio > 0.5:
            print("Sample missing paths:", missing_samples)
            raise FileNotFoundError(
                f"More than 50% of files missing in {name} dataset."
            )

    check_files_exist(df_train_check, "Train")
    check_files_exist(df_val_check, "Validation")
    check_files_exist(df_test_check, "Test")

    # 4. Verify Stratification
    print("\nVerifying Stratification...")
    train_pos_ratio = df_train_check["has_cactus"].mean()
    val_pos_ratio = df_val_check["has_cactus"].mean()

    diff = abs(train_pos_ratio - val_pos_ratio)
    print(f"Train Positive Ratio: {train_pos_ratio:.4f}")
    print(f"Val Positive Ratio: {val_pos_ratio:.4f}")
    print(f"Difference: {diff:.4f}")

    # Allow a small tolerance for stratification differences due to discrete counts
    if diff > 0.01:
        raise AssertionError(
            f"Stratification failed. Difference between train and val ratios is {diff:.4f}, expected < 0.01"
        )

    print("\nMetadata generation and verification completed successfully.")


if __name__ == "__main__":
    main()
