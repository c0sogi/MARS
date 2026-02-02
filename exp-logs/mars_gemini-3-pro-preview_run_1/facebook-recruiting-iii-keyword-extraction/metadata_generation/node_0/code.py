import pandas as pd
import numpy as np
import os
import gc
from sklearn.model_selection import train_test_split


def run_metadata_generation():
    # Configuration
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    RANDOM_STATE = 42
    VAL_SIZE = 0.2

    # Ensure output directory exists
    os.makedirs(METADATA_DIR, exist_ok=True)

    print("Starting metadata generation...")

    # ---------------------------------------------------------
    # 1. Process Training Data
    # ---------------------------------------------------------
    train_file = "train.csv"
    train_path = os.path.join(INPUT_DIR, train_file)

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Input file not found: {train_path}")

    print(f"Reading {train_path}...")
    # Read only Id and Tags to save memory.
    # train.csv has columns: Id, Title, Body, Tags
    try:
        df_full = pd.read_csv(
            train_path, usecols=["Id", "Tags"], dtype={"Id": "int64", "Tags": "object"}
        )
    except ValueError as e:
        # Fallback if columns are named differently or indices are different
        print(f"Error reading columns: {e}. Reading first few lines to debug...")
        print(pd.read_csv(train_path, nrows=5))
        raise e

    print(f"Loaded {len(df_full)} rows from training data.")

    # Drop rows with missing tags (essential for training)
    initial_count = len(df_full)
    df_full.dropna(subset=["Tags"], inplace=True)
    dropped_count = initial_count - len(df_full)
    if dropped_count > 0:
        print(f"Dropped {dropped_count} rows with missing Tags.")

    # Add file_path column (relative to input directory)
    df_full["file_path"] = train_file

    # Split into Train and Validation
    # We use random split because 'Tags' is a high-cardinality multi-label string.
    # Exact stratification on the string combination is impossible for singletons.
    # With millions of samples, random split preserves distribution effectively.
    print("Splitting data into Train and Validation sets...")
    train_df, val_df = train_test_split(
        df_full, test_size=VAL_SIZE, random_state=RANDOM_STATE, shuffle=True
    )

    # Save Metadata
    print("Saving training metadata...")
    train_meta_path = os.path.join(METADATA_DIR, "train_metadata.csv")
    train_df.to_csv(train_meta_path, index=False)

    print("Saving validation metadata...")
    val_meta_path = os.path.join(METADATA_DIR, "val_metadata.csv")
    val_df.to_csv(val_meta_path, index=False)

    # Clean up memory
    del df_full, train_df, val_df
    gc.collect()

    # ---------------------------------------------------------
    # 2. Process Test Data
    # ---------------------------------------------------------
    test_file = "test.csv"
    test_path = os.path.join(INPUT_DIR, test_file)

    if os.path.exists(test_path):
        print(f"Reading {test_path}...")
        # test.csv has columns: Id, Title, Body (No Tags)
        df_test = pd.read_csv(test_path, usecols=["Id"], dtype={"Id": "int64"})

        df_test["file_path"] = test_file

        print(f"Loaded {len(df_test)} rows from test data.")
        print("Saving test metadata...")
        test_meta_path = os.path.join(METADATA_DIR, "test_metadata.csv")
        df_test.to_csv(test_meta_path, index=False)

        del df_test
        gc.collect()
    else:
        print(f"Warning: {test_path} not found. Skipping test metadata.")

    # ---------------------------------------------------------
    # 3. Validation and Checks
    # ---------------------------------------------------------
    print("\n--- Performing Validation Checks ---")

    # Load generated metadata
    meta_train = pd.read_csv(os.path.join(METADATA_DIR, "train_metadata.csv"))
    meta_val = pd.read_csv(os.path.join(METADATA_DIR, "val_metadata.csv"))
    meta_test = pd.read_csv(os.path.join(METADATA_DIR, "test_metadata.csv"))

    # 3.1 Summary Statistics
    print(f"Train Set Shape: {meta_train.shape}")
    print(f"Val Set Shape:   {meta_val.shape}")
    print(f"Test Set Shape:  {meta_test.shape}")

    # 3.2 Check Split Ratio
    total_train_val = len(meta_train) + len(meta_val)
    actual_val_ratio = len(meta_val) / total_train_val
    print(f"Actual Validation Ratio: {actual_val_ratio:.4f}")

    if abs(actual_val_ratio - VAL_SIZE) > 0.01:
        raise AssertionError(
            f"Validation split ratio {actual_val_ratio:.4f} deviates from target {VAL_SIZE}"
        )

    # 3.3 Check File Paths
    print("Checking file path resolution...")
    datasets = {"Train": meta_train, "Validation": meta_val, "Test": meta_test}

    for name, df in datasets.items():
        if df.empty:
            continue

        # Randomly sample 1000 paths
        sample_size = min(1000, len(df))
        sample = df.sample(n=sample_size, random_state=RANDOM_STATE)

        missing_count = 0
        missing_examples = []

        for _, row in sample.iterrows():
            # Construct full path relative to current working dir
            # Metadata contains path relative to ./input, but since we just put filename there
            # we join INPUT_DIR with the file_path column.
            rel_path = row["file_path"]
            full_path = os.path.join(INPUT_DIR, rel_path)

            if not os.path.exists(full_path):
                missing_count += 1
                if len(missing_examples) < 5:
                    missing_examples.append(full_path)

        missing_ratio = missing_count / sample_size
        print(f"[{name}] Missing File Ratio: {missing_ratio:.4f}")

        if missing_ratio > 0.5:
            print(f"Examples of missing paths: {missing_examples}")
            raise FileNotFoundError(
                f"More than 50% of file paths in {name} metadata do not resolve."
            )

    # 3.4 Verify Distribution (Stratification Check)
    print("Verifying label distribution between Train and Validation...")

    def get_tag_distribution(df, top_n=20):
        # Split tags by space and explode to get individual tag counts
        all_tags = df["Tags"].astype(str).str.split().explode()
        return all_tags.value_counts(normalize=True).head(top_n)

    train_dist = get_tag_distribution(meta_train)
    val_dist = get_tag_distribution(meta_val)

    print("\nTop 5 Tags (Train):")
    print(train_dist.head(5))
    print("\nTop 5 Tags (Validation):")
    print(val_dist.head(5))

    # Check if the most frequent tag is the same and has similar frequency
    top_tag_train = train_dist.index[0]
    top_tag_val = val_dist.index[0]

    # Assert that the distributions are somewhat aligned
    # We check if the top tag frequency is within 10% relative error or 1% absolute error
    freq_train = train_dist.iloc[0]
    # Find frequency of that tag in val
    freq_val = val_dist.get(top_tag_train, 0)

    diff = abs(freq_train - freq_val)
    print(f"\nFrequency difference for top tag '{top_tag_train}': {diff:.4f}")

    if diff > 0.05:  # Allow 5% absolute difference
        raise AssertionError(
            "Significant distribution mismatch detected between Train and Validation sets."
        )

    print("\nMetadata generation and validation completed successfully.")


if __name__ == "__main__":
    run_metadata_generation()
