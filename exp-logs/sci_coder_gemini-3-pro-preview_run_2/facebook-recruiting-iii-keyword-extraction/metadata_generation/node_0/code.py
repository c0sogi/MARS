import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split


def main():
    # Define directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Ensure metadata directory exists
    os.makedirs(METADATA_DIR, exist_ok=True)

    # Define file paths
    raw_train_path = os.path.join(INPUT_DIR, "train.csv")
    raw_test_path = os.path.join(INPUT_DIR, "test.csv")

    # ---------------------------------------------------------
    # 1. Process Training Data
    # ---------------------------------------------------------
    print(f"Loading {raw_train_path}...")
    try:
        # Load data. engine='c' is faster.
        df_train_full = pd.read_csv(raw_train_path, engine="c")
    except Exception as e:
        print(f"Error loading train.csv: {e}")
        sys.exit(1)

    print(f"Original Train Shape: {df_train_full.shape}")

    # Prepare for Stratified Split
    # We stratify on the first tag to maintain topic distribution.
    # Ensure Tags is string
    df_train_full["Tags"] = df_train_full["Tags"].fillna("unknown").astype(str)

    # Extract primary tag (first tag in the list)
    def get_primary_tag(tags):
        clean_tags = tags.strip()
        if not clean_tags:
            return "unknown"
        return clean_tags.split()[0]

    primary_tags = df_train_full["Tags"].apply(get_primary_tag)

    # Handle rare classes: classes with < 2 samples cannot be stratified in a single split
    tag_counts = primary_tags.value_counts()
    rare_tags = tag_counts[tag_counts < 2].index

    # Replace rare tags with 'other' for the purpose of stratification
    stratify_col = primary_tags.apply(lambda x: "other" if x in rare_tags else x)

    print("Splitting data (80% Train, 20% Validation) with Stratification...")
    train_df, val_df = train_test_split(
        df_train_full,
        test_size=0.2,
        random_state=42,
        stratify=stratify_col,
        shuffle=True,
    )

    # Add source_file metadata (relative to ./input directory)
    train_df["source_file"] = "train.csv"
    val_df["source_file"] = "train.csv"

    # Save metadata files
    print("Saving train and validation metadata...")
    train_df.to_csv(os.path.join(METADATA_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(METADATA_DIR, "validation.csv"), index=False)

    # Cleanup memory
    del df_train_full, train_df, val_df, primary_tags, stratify_col
    import gc

    gc.collect()

    # ---------------------------------------------------------
    # 2. Process Test Data
    # ---------------------------------------------------------
    print(f"Loading {raw_test_path}...")
    try:
        df_test = pd.read_csv(raw_test_path, engine="c")
    except Exception as e:
        print(f"Error loading test.csv: {e}")
        sys.exit(1)

    print(f"Original Test Shape: {df_test.shape}")

    # Add source_file metadata
    df_test["source_file"] = "test.csv"

    # Save metadata file
    print("Saving test metadata...")
    df_test.to_csv(os.path.join(METADATA_DIR, "test.csv"), index=False)

    del df_test
    gc.collect()

    # ---------------------------------------------------------
    # 3. Validation Checks
    # ---------------------------------------------------------
    print("\nPerforming validation checks...")

    # Load generated metadata
    meta_train = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
    meta_val = pd.read_csv(os.path.join(METADATA_DIR, "validation.csv"))
    meta_test = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))

    # A. Summary Statistics
    print("\n=== Dataset Summaries ===")
    print(f"Train Set:      {len(meta_train)} samples")
    print(f"Validation Set: {len(meta_val)} samples")
    print(f"Test Set:       {len(meta_test)} samples")

    # Check Class Distribution (Top 5)
    print("\nTop 5 Tags in Train:")
    print(
        meta_train["Tags"]
        .astype(str)
        .apply(lambda x: x.split()[0])
        .value_counts()
        .head()
    )

    print("\nTop 5 Tags in Validation:")
    print(
        meta_val["Tags"].astype(str).apply(lambda x: x.split()[0]).value_counts().head()
    )

    # B. File Path Check
    def validate_paths(df, name):
        if "source_file" not in df.columns:
            return

        # Check 1000 random paths
        n_check = min(1000, len(df))
        samples = df["source_file"].sample(n=n_check, random_state=42).tolist()

        missing = 0
        missing_examples = []

        for rel_path in samples:
            # Path in metadata is relative to ./input
            full_path = os.path.join(INPUT_DIR, rel_path)
            if not os.path.exists(full_path):
                missing += 1
                if len(missing_examples) < 5:
                    missing_examples.append(rel_path)

        ratio = missing / n_check
        print(f"\nMissing file ratio for {name}: {ratio:.4f}")

        if ratio > 0.5:
            print(f"Example missing paths: {missing_examples}")
            raise RuntimeError(
                f"Validation Failed: Missing file ratio {ratio} > 0.5 for {name}"
            )

    validate_paths(meta_train, "Train")
    validate_paths(meta_val, "Validation")
    validate_paths(meta_test, "Test")

    # C. Verify Split Requirements
    # 1. Check Ratio
    total_train = len(meta_train)
    total_val = len(meta_val)
    total = total_train + total_val
    val_ratio = total_val / total

    print(f"\nValidation Split Ratio: {val_ratio:.4f}")
    if not (0.19 <= val_ratio <= 0.21):
        raise AssertionError(
            f"Validation split ratio {val_ratio:.4f} is not close to 0.2"
        )

    # 2. Check Overlap
    train_ids = set(meta_train["Id"])
    val_ids = set(meta_val["Id"])
    overlap = train_ids.intersection(val_ids)

    if len(overlap) > 0:
        raise AssertionError(
            f"Data Leakage Detected: {len(overlap)} IDs found in both train and validation sets."
        )

    print("\nAll validation checks passed successfully.")


if __name__ == "__main__":
    main()
