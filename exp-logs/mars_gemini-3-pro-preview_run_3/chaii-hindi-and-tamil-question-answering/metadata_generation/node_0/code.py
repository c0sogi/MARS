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
    train_path = os.path.join(INPUT_DIR, "train.csv")
    test_path = os.path.join(INPUT_DIR, "test.csv")

    # Load datasets
    # Using 'on_bad_lines' to ensure robustness, though data is expected to be clean
    df_train_full = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    print(f"Original Training Data Shape: {df_train_full.shape}")
    print(f"Test Data Shape: {df_test.shape}")

    # ---------------------------------------------------------
    # Split Training Data
    # ---------------------------------------------------------
    # We stratify by 'language' to ensure balanced representation in train and val
    print(
        f"Splitting training data with validation size {VAL_SIZE} and stratification by 'language'..."
    )

    train_df, val_df = train_test_split(
        df_train_full,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=df_train_full["language"],
    )

    # ---------------------------------------------------------
    # Save Metadata
    # ---------------------------------------------------------
    # For this text task, the metadata contains the text itself.
    # We save as CSV in the metadata folder.

    train_meta_path = os.path.join(METADATA_DIR, "train.csv")
    val_meta_path = os.path.join(METADATA_DIR, "val.csv")
    test_meta_path = os.path.join(METADATA_DIR, "test.csv")

    print("Saving metadata files...")
    train_df.to_csv(train_meta_path, index=False)
    val_df.to_csv(val_meta_path, index=False)
    df_test.to_csv(test_meta_path, index=False)

    print("Metadata generation complete.")

    # ---------------------------------------------------------
    # Verification and Checks
    # ---------------------------------------------------------
    print("\nStarting verification checks...")

    # Reload datasets
    df_train_new = pd.read_csv(train_meta_path)
    df_val_new = pd.read_csv(val_meta_path)
    df_test_new = pd.read_csv(test_meta_path)

    # 1. Summary Statistics
    print("\n--- Summary Statistics ---")
    print(f"Train Set: {len(df_train_new)} samples")
    print(f"Val Set:   {len(df_val_new)} samples")
    print(f"Test Set:  {len(df_test_new)} samples")

    print("\nTrain Language Distribution:")
    print(df_train_new["language"].value_counts(normalize=True))
    print("\nVal Language Distribution:")
    print(df_val_new["language"].value_counts(normalize=True))

    # 2. File Path Check
    # The dataset does not contain paths to external media files (images/audio),
    # so we skip the "missing file ratio" check logic for external assets.
    # However, we verify that the metadata files themselves were created successfully.
    assert os.path.exists(train_meta_path), "Metadata file train.csv missing"
    assert os.path.exists(val_meta_path), "Metadata file val.csv missing"
    assert os.path.exists(test_meta_path), "Metadata file test.csv missing"

    # 3. Verify Split Requirements
    print("\nVerifying split constraints...")

    # Check Split Ratio
    total_train_val = len(df_train_new) + len(df_val_new)
    actual_val_ratio = len(df_val_new) / total_train_val
    print(f"Actual Validation Ratio: {actual_val_ratio:.4f} (Target: {VAL_SIZE})")

    # Allow a small margin of error due to integer division
    assert (
        abs(actual_val_ratio - VAL_SIZE) < 0.01
    ), f"Validation split ratio mismatch. Expected {VAL_SIZE}, got {actual_val_ratio:.4f}"

    # Check Stratification
    # We check if the distribution of languages is roughly the same in train and val
    train_dist = df_train_new["language"].value_counts(normalize=True)
    val_dist = df_val_new["language"].value_counts(normalize=True)

    for lang in train_dist.index:
        diff = abs(train_dist[lang] - val_dist.get(lang, 0))
        print(f"Language '{lang}' difference: {diff:.6f}")
        assert (
            diff < 0.01
        ), f"Stratification failed for language {lang}. Train: {train_dist[lang]}, Val: {val_dist.get(lang, 0)}"

    print("\nAll checks passed successfully.")


if __name__ == "__main__":
    main()
