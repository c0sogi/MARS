import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split

# Configuration
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
RANDOM_STATE = 42
TRAIN_RATIO = 0.8
VAL_RATIO = 0.2


def main():
    # 1. Setup Directories
    if not os.path.exists(METADATA_DIR):
        os.makedirs(METADATA_DIR)

    # 2. Load Data
    print("Loading datasets...")
    try:
        train_df = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))
        sample_sub_df = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv"))
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    # 3. Preprocess and Add File Paths
    # Paths must be relative to ./input
    train_df["file_path"] = "train_images/" + train_df["image"]

    # For test data, we use sample_submission to define the test set
    test_df = sample_sub_df.copy()
    test_df["file_path"] = "test_images/" + test_df["image"]

    # 4. Stratified Split
    print("Splitting training data...")

    # Analyze label distribution for stratification
    # We treat the label combination string as the class for stratification
    label_counts = train_df["labels"].value_counts()

    # Identify classes with only 1 sample (cannot be stratified)
    rare_labels = label_counts[label_counts < 2].index.tolist()

    if rare_labels:
        print(f"Found {len(rare_labels)} rare label combinations with only 1 sample.")
        # Separate rare and common samples
        mask_rare = train_df["labels"].isin(rare_labels)
        df_rare = train_df[mask_rare]
        df_common = train_df[~mask_rare]

        # Stratify split the common samples
        train_common, val_common = train_test_split(
            df_common,
            test_size=VAL_RATIO,
            random_state=RANDOM_STATE,
            stratify=df_common["labels"],
        )

        # Add rare samples to training set to preserve them
        train_split = (
            pd.concat([train_common, df_rare])
            .sample(frac=1, random_state=RANDOM_STATE)
            .reset_index(drop=True)
        )
        val_split = val_common.reset_index(drop=True)
    else:
        # Standard stratified split
        train_split, val_split = train_test_split(
            train_df,
            test_size=VAL_RATIO,
            random_state=RANDOM_STATE,
            stratify=train_df["labels"],
        )
        train_split = train_split.reset_index(drop=True)
        val_split = val_split.reset_index(drop=True)

    # 5. Save Metadata
    print("Saving metadata...")
    cols_to_save = ["image", "labels", "file_path"]

    train_meta_path = os.path.join(METADATA_DIR, "train.csv")
    val_meta_path = os.path.join(METADATA_DIR, "val.csv")
    test_meta_path = os.path.join(METADATA_DIR, "test.csv")

    train_split[cols_to_save].to_csv(train_meta_path, index=False)
    val_split[cols_to_save].to_csv(val_meta_path, index=False)
    test_df[cols_to_save].to_csv(test_meta_path, index=False)

    # 6. Validation and Checks
    print("\nPerforming validation checks...")

    # Reload to verify saving worked
    df_train_loaded = pd.read_csv(train_meta_path)
    df_val_loaded = pd.read_csv(val_meta_path)
    df_test_loaded = pd.read_csv(test_meta_path)

    # 6a. Print Summary Statistics
    print("\n--- Summary Statistics ---")
    print(f"Train Set: {len(df_train_loaded)} samples")
    print(f"Val Set:   {len(df_val_loaded)} samples")
    print(f"Test Set:  {len(df_test_loaded)} samples")

    print("\nTrain Label Distribution (Top 5):")
    print(df_train_loaded["labels"].value_counts(normalize=True).head(5))

    print("\nVal Label Distribution (Top 5):")
    print(df_val_loaded["labels"].value_counts(normalize=True).head(5))

    # 6b. Check File Paths
    check_file_paths(df_train_loaded, "Train")
    check_file_paths(df_val_loaded, "Val")
    check_file_paths(df_test_loaded, "Test")

    # 6c. Verify Stratification
    verify_stratification(df_train_loaded, df_val_loaded)

    print("\nAll checks passed successfully.")


def check_file_paths(df, name):
    """Checks if a random sample of file paths exist."""
    print(f"\nChecking {name} file paths...")

    # Sample 1000 or all if less than 1000
    n_samples = min(1000, len(df))
    sample = df.sample(n=n_samples, random_state=RANDOM_STATE)

    missing_count = 0
    missing_samples = []

    for _, row in sample.iterrows():
        # Path is relative to ./input
        full_path = os.path.join(INPUT_DIR, row["file_path"])
        if not os.path.exists(full_path):
            missing_count += 1
            if len(missing_samples) < 5:
                missing_samples.append(row["file_path"])

    missing_ratio = missing_count / n_samples
    print(f"Missing ratio: {missing_ratio:.4f} ({missing_count}/{n_samples})")

    if missing_ratio > 0.5:
        print("Sample missing paths:", missing_samples)
        raise FileNotFoundError(
            f"Too many missing files in {name} dataset! Ratio: {missing_ratio}"
        )


def verify_stratification(train_df, val_df):
    """Verifies that the label distribution is similar between train and val."""
    print("\nVerifying stratification...")

    # Calculate normalized value counts
    train_dist = train_df["labels"].value_counts(normalize=True)
    val_dist = val_df["labels"].value_counts(normalize=True)

    # Align indices (some labels might be in train but not val if they were rare)
    all_labels = set(train_dist.index) | set(val_dist.index)

    max_diff = 0
    label_max_diff = ""

    for label in all_labels:
        train_prop = train_dist.get(label, 0)
        val_prop = val_dist.get(label, 0)
        diff = abs(train_prop - val_prop)

        if diff > max_diff:
            max_diff = diff
            label_max_diff = label

    print(f"Maximum distribution difference: {max_diff:.4f} (Label: {label_max_diff})")

    # Assert that the split is reasonably stratified.
    # We allow some deviation, especially for rare classes or small validation sets.
    # For a dataset of ~15k, 20% is 3k. A difference of > 0.05 (5%) would be significant for major classes.
    if max_diff > 0.05:
        # Check if the label causing the diff is a rare one (which we forced into train)
        # If it's a common label, this is an error.
        raise AssertionError(
            f"Stratification failed! Label '{label_max_diff}' differs by {max_diff:.4f} between train and val."
        )

    print("Stratification verification passed.")


if __name__ == "__main__":
    main()
