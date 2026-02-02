import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
import sys


def main():
    # --- Configuration ---
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    TRAIN_DIR = os.path.join(INPUT_DIR, "train")
    TEST_DIR = os.path.join(INPUT_DIR, "test")
    RANDOM_STATE = 42

    # Ensure metadata directory exists
    os.makedirs(METADATA_DIR, exist_ok=True)

    print("Starting metadata generation...")

    # --- Helper: Map IDs to Filenames ---
    # Scans directory to find exact filenames for IDs
    def get_id_map(directory):
        id_map = {}
        if not os.path.exists(directory):
            return id_map
        with os.scandir(directory) as entries:
            for entry in entries:
                if entry.is_file():
                    # Assuming ID is the filename without extension
                    file_id = os.path.splitext(entry.name)[0]
                    id_map[file_id] = entry.name
        return id_map

    print(f"Scanning {TRAIN_DIR}...")
    train_file_map = get_id_map(TRAIN_DIR)
    print(f"Found {len(train_file_map)} training files.")

    print(f"Scanning {TEST_DIR}...")
    test_file_map = get_id_map(TEST_DIR)
    print(f"Found {len(test_file_map)} test files.")

    # --- 1. Process Training Data ---
    train_csv_path = os.path.join(INPUT_DIR, "train.csv")
    if not os.path.exists(train_csv_path):
        raise FileNotFoundError(f"{train_csv_path} not found.")

    df_train_full = pd.read_csv(train_csv_path)

    # Resolve file paths
    def get_rel_path(file_id, folder_name, mapping):
        fname = mapping.get(file_id)
        if fname:
            return os.path.join(folder_name, fname)
        return None

    df_train_full["file_path"] = df_train_full["id"].apply(
        lambda x: get_rel_path(x, "train", train_file_map)
    )

    # Filter out missing files
    initial_count = len(df_train_full)
    df_train_full = df_train_full.dropna(subset=["file_path"])
    final_count = len(df_train_full)
    if initial_count != final_count:
        print(
            f"Warning: Dropped {initial_count - final_count} training rows due to missing files."
        )

    # Split Train/Validation
    # Using random split with stratification proxy (label count) or just random.
    # Given 120k samples, random split is robust.
    # We use random_state=42 as required.
    print("Splitting training data into Train/Val (80:20)...")
    train_df, val_df = train_test_split(
        df_train_full, test_size=0.2, random_state=RANDOM_STATE, shuffle=True
    )

    # --- 2. Process Test Data ---
    # We use sample_submission.csv to get the list of test IDs
    sample_sub_path = os.path.join(INPUT_DIR, "sample_submission.csv")
    if os.path.exists(sample_sub_path):
        df_test = pd.read_csv(sample_sub_path)
        # We only need 'id'
        df_test = df_test[["id"]].copy()
    else:
        # Fallback: List all files in test dir if sample_submission is missing
        print("sample_submission.csv not found, using file list for test set.")
        df_test = pd.DataFrame({"id": list(test_file_map.keys())})

    df_test["file_path"] = df_test["id"].apply(
        lambda x: get_rel_path(x, "test", test_file_map)
    )

    # Filter missing test files (optional, but good for cleanliness)
    df_test = df_test.dropna(subset=["file_path"])

    # --- 3. Save Metadata ---
    train_meta_path = os.path.join(METADATA_DIR, "train_metadata.csv")
    val_meta_path = os.path.join(METADATA_DIR, "val_metadata.csv")
    test_meta_path = os.path.join(METADATA_DIR, "test_metadata.csv")

    train_df.to_csv(train_meta_path, index=False)
    val_df.to_csv(val_meta_path, index=False)
    df_test.to_csv(test_meta_path, index=False)

    print("Metadata files saved.")

    # --- 4. Validation and Checks ---
    print("\n--- Performing Validation Checks ---")

    def check_dataset(name, df_path, expect_labels=False):
        print(f"Checking {name} dataset...")
        df = pd.read_csv(df_path)

        # Stats
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")

        # Check for labels
        if expect_labels:
            if "attribute_ids" not in df.columns:
                raise ValueError(f"{name} metadata missing 'attribute_ids' column.")
            print(f"  Unique labels (approx): {df['attribute_ids'].nunique()}")

        # Check File Paths (Random Sample of 1000)
        sample_size = min(1000, len(df))
        if sample_size > 0:
            sample_paths = (
                df["file_path"]
                .sample(n=sample_size, random_state=RANDOM_STATE)
                .tolist()
            )
            missing_count = 0
            missing_examples = []

            for p in sample_paths:
                full_path = os.path.join(INPUT_DIR, p)
                if not os.path.exists(full_path):
                    missing_count += 1
                    if len(missing_examples) < 5:
                        missing_examples.append(p)

            ratio = missing_count / sample_size
            print(f"  Missing file ratio: {ratio:.4f}")

            if ratio > 0.5:
                print(f"  Example missing paths: {missing_examples}")
                raise FileNotFoundError(
                    f"Validation failed: Too many missing files in {name} metadata."
                )

        return df

    # Load and check
    df_train_check = check_dataset("Train", train_meta_path, expect_labels=True)
    df_val_check = check_dataset("Validation", val_meta_path, expect_labels=True)
    df_test_check = check_dataset("Test", test_meta_path, expect_labels=False)

    # --- 5. Verify Stratification ---
    print("\n--- Verifying Stratification ---")

    def get_label_distribution(df):
        # Parse space-separated attribute_ids
        all_labels = []
        for ids_str in df["attribute_ids"]:
            if isinstance(ids_str, str):
                all_labels.extend(ids_str.split())
        return Counter(all_labels)

    train_counts = get_label_distribution(df_train_check)
    val_counts = get_label_distribution(df_val_check)

    total_train_labels = sum(train_counts.values())
    total_val_labels = sum(val_counts.values())

    if total_train_labels == 0 or total_val_labels == 0:
        print("Warning: No labels found to verify stratification.")
    else:
        # Check top 20 most common labels
        top_labels = [k for k, v in train_counts.most_common(20)]
        print(f"{'Label':<10} {'Train Freq':<12} {'Val Freq':<12} {'Diff':<10}")

        max_diff = 0.0
        for label in top_labels:
            train_freq = train_counts[label] / total_train_labels
            val_freq = val_counts[label] / total_val_labels
            diff = abs(train_freq - val_freq)
            max_diff = max(max_diff, diff)
            print(f"{label:<10} {train_freq:.4f}       {val_freq:.4f}       {diff:.4f}")

        print(f"Max difference in top 20 labels: {max_diff:.4f}")

        # Assert that the split is reasonably stratified
        # For 120k samples, random split should be very close.
        # We allow a small tolerance (e.g., 1-2% difference is acceptable for multi-label tails,
        # but for top classes it should be very low).
        if max_diff > 0.05:
            raise AssertionError(
                "Stratification check failed: Significant difference in label distribution between Train and Val."
            )
        else:
            print("Stratification verification passed.")

    print("\nScript completed successfully.")


if __name__ == "__main__":
    main()
