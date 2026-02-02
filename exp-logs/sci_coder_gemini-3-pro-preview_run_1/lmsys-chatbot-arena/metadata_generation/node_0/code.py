import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def generate_metadata():
    # Define paths
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    TRAIN_PATH = os.path.join(INPUT_DIR, "train.csv")
    TEST_PATH = os.path.join(INPUT_DIR, "test.csv")

    # Ensure metadata directory exists
    os.makedirs(METADATA_DIR, exist_ok=True)

    # Load raw data
    print("Loading raw data...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    # --- Preprocessing for Stratification ---
    # The target is one of three columns: winner_model_a, winner_model_b, winner_tie.
    # We convert this to a single label for stratified splitting.
    # We assume the columns are binary or probabilities summing to 1.
    # We use argmax to determine the primary class.

    label_cols = ["winner_model_a", "winner_model_b", "winner_tie"]

    # Check if columns exist
    if not all(col in train_df.columns for col in label_cols):
        raise ValueError(
            f"Training data missing one of the target columns: {label_cols}"
        )

    # Create a temporary 'target_class' column for stratification
    # 0: model_a, 1: model_b, 2: tie
    train_df["target_class"] = train_df[label_cols].idxmax(axis=1)

    # --- Create Validation Split ---
    print("Splitting data into training and validation sets...")
    RANDOM_STATE = 42
    TEST_SIZE = 0.2

    # Stratified split
    train_split, val_split = train_test_split(
        train_df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=train_df["target_class"],
    )

    # Drop the temporary column used for stratification
    train_split = train_split.drop(columns=["target_class"])
    val_split = val_split.drop(columns=["target_class"])
    train_df = train_df.drop(columns=["target_class"])  # Clean up original df

    # --- Save Metadata ---
    print("Saving metadata files...")
    train_meta_path = os.path.join(METADATA_DIR, "train_metadata.csv")
    val_meta_path = os.path.join(METADATA_DIR, "val_metadata.csv")
    test_meta_path = os.path.join(METADATA_DIR, "test_metadata.csv")

    train_split.to_csv(train_meta_path, index=False)
    val_split.to_csv(val_meta_path, index=False)
    test_df.to_csv(test_meta_path, index=False)

    print(f"Metadata saved to {METADATA_DIR}")

    # --- Verification Step ---
    print("\n--- Verifying Generated Metadata ---")

    # Reload datasets
    df_train_new = pd.read_csv(train_meta_path)
    df_val_new = pd.read_csv(val_meta_path)
    df_test_new = pd.read_csv(test_meta_path)

    # 1. Summary Statistics
    print("\nSummary Statistics:")
    print(f"Train set shape: {df_train_new.shape}")
    print(f"Val set shape:   {df_val_new.shape}")
    print(f"Test set shape:  {df_test_new.shape}")

    # Calculate class distributions
    def get_distribution(df, cols):
        # Re-derive class for stats
        classes = df[cols].idxmax(axis=1)
        return classes.value_counts(normalize=True).sort_index()

    print("\nClass Distribution (Train):")
    train_dist = get_distribution(df_train_new, label_cols)
    print(train_dist)

    print("\nClass Distribution (Validation):")
    val_dist = get_distribution(df_val_new, label_cols)
    print(val_dist)

    # 2. Check File Paths (If applicable)
    # In this task, the data is text within the CSV, not external files.
    # However, we will implement the logic as requested.
    # We check if columns looking like paths exist.
    path_columns = [
        col
        for col in df_train_new.columns
        if "path" in col.lower() or "file" in col.lower()
    ]

    if path_columns:
        print(f"\nChecking file paths in columns: {path_columns}")
        # Combine all dfs for checking
        all_dfs = [df_train_new, df_val_new, df_test_new]

        for df in all_dfs:
            for col in path_columns:
                # Sample 1000 paths
                sample_paths = (
                    df[col]
                    .dropna()
                    .sample(n=min(1000, len(df)), random_state=42)
                    .tolist()
                )
                missing_count = 0
                missing_samples = []

                for path in sample_paths:
                    # Paths are relative to ./input
                    full_path = os.path.join(INPUT_DIR, str(path))
                    if not os.path.exists(full_path):
                        missing_count += 1
                        if len(missing_samples) < 5:
                            missing_samples.append(path)

                missing_ratio = missing_count / len(sample_paths) if sample_paths else 0
                print(f"Column '{col}': Missing ratio = {missing_ratio:.4f}")

                if missing_ratio > 0.5:
                    print("Sample missing paths:", missing_samples)
                    raise FileNotFoundError(
                        f"More than 50% of files missing in column {col}"
                    )
    else:
        print(
            "\nNo explicit file path columns found to verify. Skipping file existence check."
        )

    # 3. Verify Stratification
    print("\nVerifying Stratification...")
    # We compare the proportions of each class in Train vs Val
    # We expect them to be very close.

    # Align indices for comparison
    train_dist_aligned, val_dist_aligned = train_dist.align(val_dist, fill_value=0)

    diff = (train_dist_aligned - val_dist_aligned).abs()
    max_diff = diff.max()

    print(
        f"Maximum difference in class proportions between Train and Val: {max_diff:.4f}"
    )

    # Assert that the difference is small (e.g., < 1%)
    if max_diff > 0.01:
        raise AssertionError(
            f"Stratification failed! Class distribution difference {max_diff:.4f} exceeds tolerance."
        )

    print("Stratification verification passed.")
    print("Metadata generation complete.")


if __name__ == "__main__":
    generate_metadata()
