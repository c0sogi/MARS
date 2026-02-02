import os
import shutil
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.model_selection import iterative_train_test_split
from collections import Counter


def generate_metadata():
    # Constants
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    RANDOM_STATE = 42

    # Clean and create metadata directory
    if os.path.exists(METADATA_DIR):
        shutil.rmtree(METADATA_DIR)
    os.makedirs(METADATA_DIR)

    print("Loading raw data...")
    train_df = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))
    sample_sub = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv"))

    # --- Prepare Training and Validation Data ---
    print("Preparing stratified split for multi-label data...")

    # Parse attribute_ids into lists
    # Ensure they are strings first, handle potential NaNs by converting to empty string if needed
    train_df["attribute_list"] = (
        train_df["attribute_ids"]
        .astype(str)
        .apply(lambda x: x.split() if x.lower() != "nan" else [])
    )

    # Binarize labels for stratification
    # Using sparse_output=True is crucial for memory efficiency with 3474 classes and 120k samples
    mlb = MultiLabelBinarizer(sparse_output=True)
    y = mlb.fit_transform(train_df["attribute_list"])

    # We split the indices of the dataframe
    X = train_df.index.values.reshape(-1, 1)

    print(f"Running iterative stratified split on {len(train_df)} samples...")
    # iterative_train_test_split attempts to maintain label distribution in both sets
    X_train, _, X_val, _ = iterative_train_test_split(X, y, test_size=0.2)

    train_indices = X_train.flatten()
    val_indices = X_val.flatten()

    # Create metadata DataFrames
    train_meta = train_df.iloc[train_indices].copy()
    val_meta = train_df.iloc[val_indices].copy()

    # Add file paths (relative to input directory)
    # Based on task description, filenames are id + .png, located in 'train' folder for training data
    train_meta["file_path"] = train_meta["id"].apply(
        lambda x: os.path.join("train", f"{x}.png")
    )
    val_meta["file_path"] = val_meta["id"].apply(
        lambda x: os.path.join("train", f"{x}.png")
    )

    # Cleanup temporary columns
    train_meta.drop(columns=["attribute_list"], inplace=True)
    val_meta.drop(columns=["attribute_list"], inplace=True)

    # --- Prepare Test Data ---
    print("Preparing test metadata...")
    test_meta = sample_sub.copy()
    # Test images are in 'test' folder
    test_meta["file_path"] = test_meta["id"].apply(
        lambda x: os.path.join("test", f"{x}.png")
    )

    # --- Save Metadata ---
    print("Saving metadata files...")
    train_meta.to_csv(os.path.join(METADATA_DIR, "train.csv"), index=False)
    val_meta.to_csv(os.path.join(METADATA_DIR, "val.csv"), index=False)
    test_meta.to_csv(os.path.join(METADATA_DIR, "test.csv"), index=False)

    return train_meta, val_meta, test_meta


def verify_metadata(train_meta, val_meta, test_meta):
    print("\n--- Verifying Metadata ---")
    INPUT_DIR = "./input"

    # 1. Summary Statistics
    print(f"Train samples: {len(train_meta)}")
    print(f"Val samples:   {len(val_meta)}")
    print(f"Test samples:  {len(test_meta)}")

    # 2. File Path Verification
    def check_paths(df, name):
        if df.empty:
            return
        # Check 1000 random paths
        sample = df.sample(n=min(1000, len(df)), random_state=42)
        missing_count = 0
        missing_examples = []

        for _, row in sample.iterrows():
            full_path = os.path.join(INPUT_DIR, row["file_path"])
            if not os.path.exists(full_path):
                missing_count += 1
                if len(missing_examples) < 5:
                    missing_examples.append(row["file_path"])

        ratio = missing_count / len(sample)
        print(f"{name} missing file ratio: {ratio:.4f}")

        if ratio > 0.5:
            print(f"Examples of missing files in {name}: {missing_examples}")
            raise FileNotFoundError(
                f"More than 50% of file paths in {name} are invalid."
            )

    print("Checking file paths...")
    check_paths(train_meta, "Train")
    check_paths(val_meta, "Val")
    check_paths(test_meta, "Test")

    # 3. Validation Split Verification
    total_train = len(train_meta) + len(val_meta)
    split_ratio = len(val_meta) / total_train
    print(f"Validation split ratio: {split_ratio:.4f}")

    # Allow some flexibility as iterative stratification prioritizes label balance over exact size
    assert (
        0.15 <= split_ratio <= 0.25
    ), f"Split ratio {split_ratio} deviates significantly from 0.2"

    # 4. Stratification Verification
    print("Verifying stratification (Top 5 labels)...")

    def get_label_distribution(df):
        all_labels = []
        # Parse strings back to lists
        lists = (
            df["attribute_ids"]
            .astype(str)
            .apply(lambda x: x.split() if x.lower() != "nan" else [])
        )
        for l in lists:
            all_labels.extend(l)
        return Counter(all_labels), len(all_labels)

    train_counts, train_total = get_label_distribution(train_meta)
    val_counts, val_total = get_label_distribution(val_meta)

    # Check top 5 most common labels
    top_labels = [k for k, v in train_counts.most_common(5)]

    for label in top_labels:
        train_freq = train_counts[label] / train_total
        val_freq = val_counts.get(label, 0) / val_total
        diff = abs(train_freq - val_freq)
        print(
            f"Label {label}: Train Freq={train_freq:.4f}, Val Freq={val_freq:.4f}, Diff={diff:.4f}"
        )

        # Assert that frequencies are reasonably close (stratification worked)
        assert (
            diff < 0.05
        ), f"Stratification failed for label {label}. Distribution mismatch."


if __name__ == "__main__":
    try:
        t_meta, v_meta, test_meta = generate_metadata()
        verify_metadata(t_meta, v_meta, test_meta)
        print("\nMetadata generation and verification completed successfully.")
    except Exception as e:
        print(f"\nExecution failed: {e}")
        raise e
