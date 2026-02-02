import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Constants
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
RANDOM_STATE = 42
VAL_SIZE = 0.2


def generate_metadata():
    # 1. Setup Directories
    os.makedirs(METADATA_DIR, exist_ok=True)

    # 2. Load Data
    print("Loading raw data...")
    train_path = os.path.join(INPUT_DIR, "train.csv")
    test_path = os.path.join(INPUT_DIR, "test.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print(f"Original Train Shape: {train_df.shape}")
    print(f"Original Test Shape: {test_df.shape}")

    # 3. Preprocess for Stratification
    # We need to determine the class label for stratification.
    # The columns are winner_model_a, winner_model_b, winner_tie.
    # We assume these are One-Hot or probabilities where argmax gives the class.
    target_cols = ["winner_model_a", "winner_model_b", "winner_tie"]

    # Create a single label column for stratification
    # idxmax returns the column name with the max value.
    # We map these names to integers for easier handling/stats.
    # If rows sum to 1 (probabilities) or are binary, this works.
    train_df["stratify_label"] = train_df[target_cols].idxmax(axis=1)

    # 4. Create Validation Split
    print("Splitting data into training and validation sets...")
    # Stratified split
    train_meta, val_meta = train_test_split(
        train_df,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=train_df["stratify_label"],
        shuffle=True,
    )

    # Drop the temporary stratification column to keep metadata clean
    train_meta = train_meta.drop(columns=["stratify_label"])
    val_meta = val_meta.drop(columns=["stratify_label"])
    # We don't drop it from train_df just yet as we might need it for verification logic if we used the original df,
    # but here we use the split dfs.

    # 5. Save Metadata
    # We save the full dataframes as metadata. This allows efficient loading downstream.
    print("Saving metadata...")
    train_meta_path = os.path.join(METADATA_DIR, "train.csv")
    val_meta_path = os.path.join(METADATA_DIR, "val.csv")
    test_meta_path = os.path.join(METADATA_DIR, "test.csv")

    train_meta.to_csv(train_meta_path, index=False)
    val_meta.to_csv(val_meta_path, index=False)
    test_df.to_csv(test_meta_path, index=False)

    print("Metadata generation complete.")
    return train_meta_path, val_meta_path, test_meta_path


def verify_metadata(train_path, val_path, test_path):
    print("\nVerifying metadata...")

    # Load generated metadata
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    # --- Summary Statistics ---
    print("\nSummary Statistics:")
    print(f"Train set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")
    print(f"Test set size: {len(test_df)}")

    target_cols = ["winner_model_a", "winner_model_b", "winner_tie"]

    # Helper to get distribution
    def get_dist(df):
        # Re-derive label for stats
        labels = df[target_cols].idxmax(axis=1)
        return labels.value_counts(normalize=True)

    train_dist = get_dist(train_df)
    val_dist = get_dist(val_df)

    print("\nClass Distribution (Train):")
    print(train_dist)
    print("\nClass Distribution (Validation):")
    print(val_dist)

    # --- Verification Checks ---

    # 1. Check Split Ratio
    total_train_val = len(train_df) + len(val_df)
    val_ratio = len(val_df) / total_train_val
    print(f"\nActual Validation Ratio: {val_ratio:.4f}")

    # Allow a tiny margin of error due to rounding/integer counts
    if not (0.19 < val_ratio < 0.21):
        raise AssertionError(
            f"Validation ratio {val_ratio} is not close to required 0.2"
        )

    # 2. Check Stratification
    # We check if the difference in proportions for any class is within a small tolerance (e.g., 1%)
    print("\nChecking stratification consistency...")
    for label in train_dist.index:
        train_prop = train_dist[label]
        val_prop = val_dist.get(label, 0)
        diff = abs(train_prop - val_prop)
        print(
            f"Label {label}: Train={train_prop:.4f}, Val={val_prop:.4f}, Diff={diff:.4f}"
        )

        if diff > 0.015:  # 1.5% tolerance
            raise AssertionError(
                f"Stratification failed for class {label}. Diff: {diff}"
            )

    # 3. File Path Check (Generic Requirement)
    # The dataset provided (Chatbot Arena) consists of text within CSVs, not external files.
    # Therefore, there are no file path columns (like 'image_path') to check against ./input.
    # We explicitly note this skip.
    print("\nChecking file paths...")
    print(
        "No external file path columns detected in metadata (Text-based CSV dataset). Skipping file existence check."
    )

    print("\nAll verification checks passed successfully.")


if __name__ == "__main__":
    try:
        t_path, v_path, te_path = generate_metadata()
        verify_metadata(t_path, v_path, te_path)
    except Exception as e:
        print(f"\nERROR: Script failed with exception: {e}")
        raise e
