import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Configuration
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
RANDOM_STATE = 42
VAL_SIZE = 0.2


def main():
    print("Starting metadata generation...")

    # Ensure metadata directory exists
    os.makedirs(METADATA_DIR, exist_ok=True)

    # Load raw data
    train_path = os.path.join(INPUT_DIR, "train.csv")
    test_path = os.path.join(INPUT_DIR, "test.csv")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train file not found at {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test file not found at {test_path}")

    df_train_full = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    print(f"Loaded raw training data: {df_train_full.shape}")
    print(f"Loaded raw test data: {df_test.shape}")

    # Perform Stratified Split
    # We stratify by 'score' to ensure the label distribution is preserved in the validation set.
    # This is appropriate as the scores are discrete levels.
    print("Splitting data (Stratified by 'score')...")
    train_df, val_df = train_test_split(
        df_train_full,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=df_train_full["score"],
    )

    # Save Metadata files
    # For this NLP task, the metadata contains the text itself.
    meta_train_path = os.path.join(METADATA_DIR, "train.csv")
    meta_val_path = os.path.join(METADATA_DIR, "val.csv")
    meta_test_path = os.path.join(METADATA_DIR, "test.csv")

    train_df.to_csv(meta_train_path, index=False)
    val_df.to_csv(meta_val_path, index=False)
    df_test.to_csv(meta_test_path, index=False)

    print(f"Saved metadata to {METADATA_DIR}")

    # --- Verification Step ---
    print("\n--- Verification ---")

    # 1. Load generated metadata
    v_train = pd.read_csv(meta_train_path)
    v_val = pd.read_csv(meta_val_path)
    v_test = pd.read_csv(meta_test_path)

    # 2. Summary Statistics
    print(f"Final Train shape: {v_train.shape}")
    print(f"Final Val shape:   {v_val.shape}")
    print(f"Final Test shape:  {v_test.shape}")

    print("\nTrain Score Distribution:")
    train_dist = v_train["score"].value_counts(normalize=True).sort_index()
    print(train_dist)

    print("\nVal Score Distribution:")
    val_dist = v_val["score"].value_counts(normalize=True).sort_index()
    print(val_dist)

    # 3. Check File Paths
    # This dataset consists of text phrases directly in the CSV.
    # There are no relative file paths to external images/audio to check.
    # We explicitly note this pass.
    print("\nFile path check: N/A (Text data contained within metadata)")

    # 4. Verify Split Requirements

    # Check Split Ratio
    total_samples = len(v_train) + len(v_val)
    actual_val_ratio = len(v_val) / total_samples
    print(f"\nActual Validation Ratio: {actual_val_ratio:.4f}")

    # Assert ratio is within small margin of error
    if abs(actual_val_ratio - VAL_SIZE) > 0.01:
        raise AssertionError(
            f"Validation split ratio mismatch. Expected {VAL_SIZE}, got {actual_val_ratio:.4f}"
        )

    # Check Stratification
    # We calculate the maximum absolute difference between class probabilities
    diff = (train_dist - val_dist).abs().max()
    print(f"Max distribution difference (Train vs Val): {diff:.6f}")

    # Assert stratification is effective (allow 1% tolerance for discrete quantization effects)
    if diff > 0.01:
        raise AssertionError(
            "Stratification failed! Label distribution differs significantly between Train and Val."
        )

    print("\nAll checks passed successfully.")


if __name__ == "__main__":
    main()
