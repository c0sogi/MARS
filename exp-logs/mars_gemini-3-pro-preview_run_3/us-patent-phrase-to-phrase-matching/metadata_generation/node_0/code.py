import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def generate_metadata():
    # Configuration
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    RANDOM_STATE = 42
    VAL_SIZE = 0.2

    # Ensure metadata directory exists
    os.makedirs(METADATA_DIR, exist_ok=True)

    print("Loading raw data...")
    train_path = os.path.join(INPUT_DIR, "train.csv")
    test_path = os.path.join(INPUT_DIR, "test.csv")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train file not found at {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test file not found at {test_path}")

    df_train_full = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    print(f"Original Training Data Shape: {df_train_full.shape}")
    print(f"Test Data Shape: {df_test.shape}")

    # Perform Stratified Split
    # We stratify by 'score' because the scores are discrete (0, 0.25, 0.5, 0.75, 1.0)
    # and we want to maintain the distribution of similarity ratings.
    print("Splitting training data into train/validation (80:20 stratified)...")

    # Check if score column exists and has discrete values suitable for stratification
    if "score" not in df_train_full.columns:
        raise ValueError("Column 'score' missing from training data.")

    # Create the split
    df_train, df_val = train_test_split(
        df_train_full,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=df_train_full["score"],
    )

    # Save metadata files
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
    print("\nStarting Verification...")

    # 1. Load the datasets
    df_train_loaded = pd.read_csv(train_meta_path)
    df_val_loaded = pd.read_csv(val_meta_path)
    df_test_loaded = pd.read_csv(test_meta_path)

    # 2. Print Summary Statistics
    print("\n--- Summary Statistics ---")
    print(f"Train Set: {len(df_train_loaded)} samples")
    print(f"Val Set:   {len(df_val_loaded)} samples")
    print(f"Test Set:  {len(df_test_loaded)} samples")

    print("\nTrain Score Distribution:")
    print(df_train_loaded["score"].value_counts(normalize=True).sort_index())

    print("\nVal Score Distribution:")
    print(df_val_loaded["score"].value_counts(normalize=True).sort_index())

    # 3. Check File Paths
    # Note: This dataset (Phrase Matching) is text-based and contained entirely within the CSVs.
    # There are no external image/audio files referenced by paths.
    # Therefore, we check if columns implying file paths exist. If not, we skip the missing file check.
    # If they did exist, we would check 1000 random paths.

    potential_path_cols = [
        c for c in df_train_loaded.columns if "path" in c.lower() or "file" in c.lower()
    ]
    if potential_path_cols:
        print(f"\nChecking file paths in columns: {potential_path_cols}")
        # Implementation of file check if paths existed (Generic handler)
        for col in potential_path_cols:
            sample_paths = (
                df_train_loaded[col]
                .sample(n=min(1000, len(df_train_loaded)), random_state=RANDOM_STATE)
                .tolist()
            )
            missing_count = 0
            missing_samples = []
            for p in sample_paths:
                # Paths in metadata must be relative to ./input
                full_p = os.path.join(INPUT_DIR, str(p))
                if not os.path.exists(full_p):
                    missing_count += 1
                    if len(missing_samples) < 5:
                        missing_samples.append(p)

            ratio = missing_count / len(sample_paths)
            print(f"Column '{col}': Missing file ratio = {ratio:.4f}")
            if ratio > 0.5:
                print("Sample missing paths:", missing_samples)
                raise FileNotFoundError(
                    f"More than 50% of files missing for column {col}"
                )
    else:
        print(
            "\nNo file path columns detected. Skipping missing file check (Data is text-only)."
        )

    # 4. Verify Validation Split Requirements
    print("\nVerifying Split Requirements...")

    # Assert split ratio
    total_train_val = len(df_train_loaded) + len(df_val_loaded)
    actual_val_ratio = len(df_val_loaded) / total_train_val
    print(f"Actual Validation Ratio: {actual_val_ratio:.4f}")

    # Allow small floating point tolerance
    if not np.isclose(actual_val_ratio, VAL_SIZE, atol=1e-3):
        raise AssertionError(
            f"Validation split ratio mismatch. Expected {VAL_SIZE}, got {actual_val_ratio}"
        )

    # Assert Stratification
    # We compare the normalized value counts of the 'score' column in Train vs Val
    train_dist = df_train_loaded["score"].value_counts(normalize=True).sort_index()
    val_dist = df_val_loaded["score"].value_counts(normalize=True).sort_index()

    # Calculate maximum difference in class proportions
    diffs = (train_dist - val_dist).abs()
    max_diff = diffs.max()
    print(
        f"Maximum difference in class proportions between Train and Val: {max_diff:.4f}"
    )

    # Tolerance for stratification mismatch (should be very low for stratified split)
    if max_diff > 0.01:  # 1% tolerance
        raise AssertionError(
            "Stratification failed. Class distributions differ significantly between Train and Val."
        )

    print("\nAll checks passed successfully.")


if __name__ == "__main__":
    generate_metadata()
