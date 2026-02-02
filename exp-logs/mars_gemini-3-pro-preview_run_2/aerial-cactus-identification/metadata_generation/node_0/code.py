import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Constants
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
RANDOM_STATE = 42
VAL_SIZE = 0.2


def main():
    # 1. Setup directories
    os.makedirs(METADATA_DIR, exist_ok=True)

    # 2. Load raw data
    train_csv_path = os.path.join(INPUT_DIR, "train.csv")
    sample_sub_path = os.path.join(INPUT_DIR, "sample_submission.csv")

    df_train_full = pd.read_csv(train_csv_path)
    df_test = pd.read_csv(sample_sub_path)

    # 3. Construct relative file paths
    # The task requires paths relative to ./input
    # Train images are in input/train/, Test images are in input/test/

    df_train_full["file_path"] = df_train_full["id"].apply(
        lambda x: os.path.join("train", x)
    )
    df_test["file_path"] = df_test["id"].apply(lambda x: os.path.join("test", x))

    # 4. Split training data into train and validation
    # Stratified split based on 'has_cactus'
    train_indices, val_indices = train_test_split(
        df_train_full.index,
        test_size=VAL_SIZE,
        stratify=df_train_full["has_cactus"],
        random_state=RANDOM_STATE,
    )

    df_train = df_train_full.loc[train_indices].copy()
    df_val = df_train_full.loc[val_indices].copy()

    # 5. Save metadata
    train_meta_path = os.path.join(METADATA_DIR, "train_metadata.csv")
    val_meta_path = os.path.join(METADATA_DIR, "val_metadata.csv")
    test_meta_path = os.path.join(METADATA_DIR, "test_metadata.csv")

    df_train.to_csv(train_meta_path, index=False)
    df_val.to_csv(val_meta_path, index=False)
    # For test, we keep the ID and file_path. The 'has_cactus' in sample_submission is a placeholder.
    # We'll keep it as is or drop it? Usually test metadata just needs IDs/paths.
    # But keeping the structure similar is fine.
    df_test.to_csv(test_meta_path, index=False)

    print("Metadata generation complete.")
    print(f"Train metadata saved to: {train_meta_path}")
    print(f"Validation metadata saved to: {val_meta_path}")
    print(f"Test metadata saved to: {test_meta_path}")

    # 6. Verification
    verify_metadata(train_meta_path, val_meta_path, test_meta_path, df_train_full)


def verify_metadata(train_path, val_path, test_path, original_train_df):
    print("\n--- Verifying Metadata ---")

    # Load datasets
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    # --- Summary Statistics ---
    print(f"Train set shape: {df_train.shape}")
    print(f"Validation set shape: {df_val.shape}")
    print(f"Test set shape: {df_test.shape}")

    print("\nTrain Class Distribution:")
    print(df_train["has_cactus"].value_counts(normalize=True))

    print("\nValidation Class Distribution:")
    print(df_val["has_cactus"].value_counts(normalize=True))

    # --- File Path Verification ---
    for name, df in [("Train", df_train), ("Validation", df_val), ("Test", df_test)]:
        print(f"\nChecking file paths for {name} set...")

        # Select random sample
        sample_size = min(1000, len(df))
        sample_paths = (
            df["file_path"].sample(n=sample_size, random_state=RANDOM_STATE).tolist()
        )

        missing_count = 0
        missing_samples = []

        for rel_path in sample_paths:
            # Resolve path relative to INPUT_DIR
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
            print("Sample missing paths:", missing_samples)
            raise FileNotFoundError(
                f"More than 50% of file paths in {name} metadata are invalid."
            )
        else:
            print(f"File path check passed for {name} set.")

    # --- Stratification Verification ---
    print("\nVerifying stratification...")
    original_ratio = original_train_df["has_cactus"].mean()
    train_ratio = df_train["has_cactus"].mean()
    val_ratio = df_val["has_cactus"].mean()

    print(f"Original positive ratio: {original_ratio:.4f}")
    print(f"Train positive ratio:    {train_ratio:.4f}")
    print(f"Val positive ratio:      {val_ratio:.4f}")

    # Check if ratios are reasonably close (within 1%)
    if (
        abs(train_ratio - original_ratio) > 0.01
        or abs(val_ratio - original_ratio) > 0.01
    ):
        raise AssertionError(
            "Stratification failed: Class distribution in split datasets differs significantly from original."
        )

    print("Stratification check passed.")
    print("\nAll checks passed successfully.")


if __name__ == "__main__":
    main()
