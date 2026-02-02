import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit


def main():
    # Define paths
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Ensure metadata directory exists
    os.makedirs(METADATA_DIR, exist_ok=True)

    # File paths based on Dataset Information
    TRAIN_CSV = os.path.join(INPUT_DIR, "en_train.csv")
    TEST_CSV = os.path.join(INPUT_DIR, "en_test_2.csv")

    print("Loading datasets...")

    # Load Training Data
    # en_train.csv contains: sentence_id, token_id, class, before, after
    # We need to handle potential quoting issues or data types, but default read_csv usually works well.
    # Specifying dtypes can help with memory and correctness.
    dtype_dict = {
        "sentence_id": "int64",
        "token_id": "int64",
        "class": "object",
        "before": "object",
        "after": "object",
    }

    if not os.path.exists(TRAIN_CSV):
        raise FileNotFoundError(f"Training file not found at {TRAIN_CSV}")

    df_train_full = pd.read_csv(TRAIN_CSV, dtype=dtype_dict)
    print(f"Loaded training data: {df_train_full.shape}")

    # Load Test Data
    if not os.path.exists(TEST_CSV):
        raise FileNotFoundError(f"Test file not found at {TEST_CSV}")

    df_test = pd.read_csv(TEST_CSV)
    print(f"Loaded test data: {df_test.shape}")

    # Create Validation Split
    # Requirement: 80:20 split, Random State 42, Group Sampling by sentence_id
    print("Splitting training data into Train/Val...")

    splitter = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)

    # We split based on sentence_id groups
    groups = df_train_full["sentence_id"]
    train_idx, val_idx = next(splitter.split(df_train_full, groups=groups))

    df_train = df_train_full.iloc[train_idx].copy()
    df_val = df_train_full.iloc[val_idx].copy()

    print(f"Train split shape: {df_train.shape}")
    print(f"Val split shape: {df_val.shape}")

    # Save Metadata (Parquet is efficient for this size)
    # We add an 'id' column to match submission format requirements (sentence_id_token_id)
    # This helps downstream models.

    def add_id_column(df):
        # Create 'id' if not present, useful for submission alignment
        if (
            "id" not in df.columns
            and "sentence_id" in df.columns
            and "token_id" in df.columns
        ):
            df["id"] = df["sentence_id"].astype(str) + "_" + df["token_id"].astype(str)
        return df

    df_train = add_id_column(df_train)
    df_val = add_id_column(df_val)
    df_test = add_id_column(df_test)

    train_meta_path = os.path.join(METADATA_DIR, "train.parquet")
    val_meta_path = os.path.join(METADATA_DIR, "val.parquet")
    test_meta_path = os.path.join(METADATA_DIR, "test.parquet")

    print("Saving metadata files...")
    df_train.to_parquet(train_meta_path, index=False)
    df_val.to_parquet(val_meta_path, index=False)
    df_test.to_parquet(test_meta_path, index=False)

    # ==========================================
    # Verification Steps
    # ==========================================
    print("\nVerifying generated metadata...")

    # 1. Load datasets using new metadata
    df_train_loaded = pd.read_parquet(train_meta_path)
    df_val_loaded = pd.read_parquet(val_meta_path)
    df_test_loaded = pd.read_parquet(test_meta_path)

    # 2. Print Summary Statistics
    def print_stats(name, df):
        print(f"\n--- {name} Statistics ---")
        print(f"Total samples: {len(df)}")
        print(f"Columns: {list(df.columns)}")
        if "class" in df.columns:
            print(f"Unique classes: {df['class'].nunique()}")
            print(f"Top 5 classes:\n{df['class'].value_counts().head(5)}")
        if "sentence_id" in df.columns:
            print(f"Unique sentences: {df['sentence_id'].nunique()}")

    print_stats("Train", df_train_loaded)
    print_stats("Validation", df_val_loaded)
    print_stats("Test", df_test_loaded)

    # 3. Check File Paths (If applicable)
    # In this task, the data is text in CSVs, not external files.
    # However, we verify the source files exist as a proxy for this requirement.
    # We also check if the metadata files we just wrote exist.
    print("\nChecking file integrity...")
    for p in [train_meta_path, val_meta_path, test_meta_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Generated metadata file missing: {p}")

    # 4. Verify Validation Split Requirements
    print("\nVerifying split requirements...")

    # Assert stratification/group split was successful
    # Check for sentence leakage: Intersection of sentence_ids should be empty
    train_sentences = set(df_train_loaded["sentence_id"].unique())
    val_sentences = set(df_val_loaded["sentence_id"].unique())

    intersection = train_sentences.intersection(val_sentences)
    if len(intersection) > 0:
        raise AssertionError(
            f"Group split failed! {len(intersection)} sentences found in both train and validation sets."
        )
    else:
        print("Group split verification passed: No sentence leakage detected.")

    # Check split ratio (approx 80:20 in terms of sentences)
    n_train_sent = len(train_sentences)
    n_val_sent = len(val_sentences)
    total_sent = n_train_sent + n_val_sent
    train_ratio = n_train_sent / total_sent

    print(f"Split Ratio (Sentences): Train={train_ratio:.4f}, Val={1-train_ratio:.4f}")

    # Allow small deviation due to group sizes, but it should be close to 0.8
    if not (0.78 < train_ratio < 0.82):
        print(
            f"Warning: Split ratio {train_ratio:.4f} deviates slightly from 0.8, likely due to large sentence groups."
        )
        # Note: We don't raise an error here as exact 80.00% is impossible with group splitting,
        # but we verified the logic used was 0.8.

    print("\nMetadata generation and verification complete.")


if __name__ == "__main__":
    main()
