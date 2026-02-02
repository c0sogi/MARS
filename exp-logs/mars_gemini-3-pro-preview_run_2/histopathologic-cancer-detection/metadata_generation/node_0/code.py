import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Constants
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
RANDOM_STATE = 42


def main():
    # Ensure metadata directory exists
    os.makedirs(METADATA_DIR, exist_ok=True)

    print("Loading raw data...")
    # Load training labels
    train_labels_path = os.path.join(INPUT_DIR, "train_labels.csv")
    df_train_full = pd.read_csv(train_labels_path)

    # Load test IDs from sample submission
    sample_submission_path = os.path.join(INPUT_DIR, "sample_submission.csv")
    df_test = pd.read_csv(sample_submission_path)

    # Construct relative file paths
    # Train images are in 'train/' and have .tif extension
    df_train_full["file_path"] = df_train_full["id"].apply(
        lambda x: os.path.join("train", f"{x}.tif")
    )

    # Test images are in 'test/' and have .tif extension
    df_test["file_path"] = df_test["id"].apply(
        lambda x: os.path.join("test", f"{x}.tif")
    )

    # Perform Stratified Split (80% Train, 20% Val)
    print("Splitting training data...")
    df_train, df_val = train_test_split(
        df_train_full,
        test_size=0.2,
        stratify=df_train_full["label"],
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    # Save metadata files
    print("Saving metadata...")
    train_meta_path = os.path.join(METADATA_DIR, "train.csv")
    val_meta_path = os.path.join(METADATA_DIR, "val.csv")
    test_meta_path = os.path.join(METADATA_DIR, "test.csv")

    df_train.to_csv(train_meta_path, index=False)
    df_val.to_csv(val_meta_path, index=False)
    # For test, we keep ID and file_path. We can keep label from sample_submission or drop it.
    # Usually test metadata just needs ID and path, but keeping format consistent is fine.
    df_test.to_csv(test_meta_path, index=False)

    print("Metadata generation complete.")

    # --- Verification Steps ---
    print("\nStarting verification...")

    # Verify each dataset
    verify_dataset(train_meta_path, "train")
    verify_dataset(val_meta_path, "validation")
    verify_dataset(test_meta_path, "test")

    # Verify split properties
    verify_split(df_train, df_val)

    print("\nAll checks passed successfully.")


def verify_dataset(meta_path, name):
    """Loads metadata, prints stats, and checks file existence."""
    print(f"\n--- Verifying {name} dataset ---")
    df = pd.read_csv(meta_path)

    # 1. Summary Statistics
    print(f"Total samples: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    if "label" in df.columns:
        print("Class distribution:")
        print(df["label"].value_counts(normalize=True))
        print(df["label"].value_counts())

    # 2. Check File Paths
    paths = df["file_path"].values
    # Select 1000 random paths (or all if less than 1000)
    n_check = min(len(paths), 1000)
    # Use a fixed seed for reproducibility of the check, though not strictly required
    rng = np.random.default_rng(seed=RANDOM_STATE)
    check_paths = rng.choice(paths, size=n_check, replace=False)

    missing_count = 0
    missing_examples = []

    for rel_path in check_paths:
        full_path = os.path.join(INPUT_DIR, rel_path)
        if not os.path.exists(full_path):
            missing_count += 1
            if len(missing_examples) < 5:
                missing_examples.append(rel_path)

    missing_ratio = missing_count / n_check
    print(
        f"File existence check ({n_check} samples): Missing Ratio = {missing_ratio:.4f}"
    )

    if missing_ratio > 0.5:
        print("Examples of missing paths:")
        for p in missing_examples:
            print(f"  {p}")
        raise FileNotFoundError(
            f"Error: More than 50% of file paths in {name} metadata are invalid."
        )


def verify_split(train_df, val_df):
    """Verifies that the split is stratified and has no overlap."""
    print("\n--- Verifying Split Requirements ---")

    # 1. Check for ID Overlap
    train_ids = set(train_df["id"])
    val_ids = set(val_df["id"])
    overlap = train_ids.intersection(val_ids)

    if overlap:
        raise AssertionError(
            f"Error: Found {len(overlap)} overlapping IDs between train and validation sets."
        )
    print("No ID overlap detected.")

    # 2. Check Stratification
    train_dist = train_df["label"].value_counts(normalize=True)
    val_dist = val_df["label"].value_counts(normalize=True)

    print("Train Label Distribution:\n", train_dist)
    print("Val Label Distribution:\n", val_dist)

    # Allow a small tolerance for stratification differences
    tolerance = 0.01
    for label in train_dist.index:
        diff = abs(train_dist[label] - val_dist[label])
        if diff > tolerance:
            raise AssertionError(
                f"Error: Stratification failed. Label {label} differs by {diff:.4f} > {tolerance}"
            )

    print("Stratification verified.")


if __name__ == "__main__":
    main()
