import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit


def main():
    # Configuration
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    TRAIN_FILE = "ru_train.csv"
    TEST_FILE = "ru_test_2.csv"
    RANDOM_STATE = 42
    VAL_SIZE = 0.2

    # Ensure metadata directory exists
    os.makedirs(METADATA_DIR, exist_ok=True)

    # Define full paths
    train_path = os.path.join(INPUT_DIR, TRAIN_FILE)
    test_path = os.path.join(INPUT_DIR, TEST_FILE)

    print("Loading raw data...")
    # Load training data
    # Specifying dtypes can help with memory, but default is usually fine for this size on 220GB RAM
    df_train_full = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    print(f"Train shape: {df_train_full.shape}")
    print(f"Test shape: {df_test.shape}")

    # Verify necessary columns for group splitting
    if "sentence_id" not in df_train_full.columns:
        # Attempt to derive sentence_id if not present but 'id' exists (format sentence_id_token_id)
        # However, description says "Each sentence has a sentence_id".
        # If strictly not present, we might fail, but let's check if we can fallback to parsing 'id'
        if "id" in df_train_full.columns:
            print("Warning: 'sentence_id' column not found. Deriving from 'id' column.")
            df_train_full["sentence_id"] = df_train_full["id"].apply(
                lambda x: x.split("_")[0]
            )
        else:
            raise ValueError(
                "Column 'sentence_id' not found in training data and cannot be derived."
            )

    # Perform Group Shuffle Split
    print("Performing Group Shuffle Split (80/20)...")
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=VAL_SIZE, random_state=RANDOM_STATE
    )

    # We split based on the groups provided by sentence_id
    train_idx, val_idx = next(
        splitter.split(df_train_full, groups=df_train_full["sentence_id"])
    )

    df_train = df_train_full.iloc[train_idx].copy()
    df_val = df_train_full.iloc[val_idx].copy()

    print(f"Split complete. Train tokens: {len(df_train)}, Val tokens: {len(df_val)}")

    # Save metadata files
    # Here, the metadata files are the split CSVs themselves containing the text data.
    train_meta_path = os.path.join(METADATA_DIR, "train.csv")
    val_meta_path = os.path.join(METADATA_DIR, "val.csv")
    test_meta_path = os.path.join(METADATA_DIR, "test.csv")

    print("Saving metadata files...")
    df_train.to_csv(train_meta_path, index=False)
    df_val.to_csv(val_meta_path, index=False)
    df_test.to_csv(test_meta_path, index=False)

    # ==========================================
    # Verification and Statistics
    # ==========================================
    print("\n--- Verifying and Generating Statistics ---")

    # Reload datasets to verify integrity
    df_train_loaded = pd.read_csv(train_meta_path)
    df_val_loaded = pd.read_csv(val_meta_path)
    df_test_loaded = pd.read_csv(test_meta_path)

    # 1. Summary Statistics
    def print_stats(name, df):
        print(f"\nDataset: {name}")
        print(f"  Total samples (tokens): {len(df)}")
        print(f"  Columns: {list(df.columns)}")
        if "sentence_id" in df.columns:
            print(f"  Unique sentences: {df['sentence_id'].nunique()}")
        if "class" in df.columns:
            print(
                f"  Class distribution:\n{df['class'].value_counts(normalize=True).head()}"
            )

    print_stats("Training", df_train_loaded)
    print_stats("Validation", df_val_loaded)
    print_stats("Test", df_test_loaded)

    # 2. File Path Check
    # The requirement asks to check relative file paths if they exist in metadata.
    # In this task, the metadata contains the text data itself, not paths to external files.
    # However, to strictly follow the structure, we can verify the input source files exist.
    print("\nChecking source file existence...")
    for p in [train_path, test_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Input file not found: {p}")
    print("Source files verified.")

    # 3. Verify Validation Split Requirements
    print("\nVerifying split requirements...")

    # Assert that the split ratio is roughly correct (in terms of groups/sentences, not necessarily tokens)
    # Note: Token count ratio might vary if sentence lengths vary, but should be close.
    train_sentences = set(df_train_loaded["sentence_id"])
    val_sentences = set(df_val_loaded["sentence_id"])

    n_train_sent = len(train_sentences)
    n_val_sent = len(val_sentences)
    total_sent = n_train_sent + n_val_sent

    actual_val_ratio = n_val_sent / total_sent
    print(f"Actual validation ratio (by sentence count): {actual_val_ratio:.4f}")

    # Allow small deviation due to group granularity
    if not (0.18 < actual_val_ratio < 0.22):
        print(
            "Warning: Validation ratio deviates slightly from 0.2, likely due to large groups."
        )
        # Not raising error here as GroupShuffleSplit is randomized, but it should be close.

    # Assert that the group split was successful (No sentence leakage)
    # Intersection of sentence_ids must be empty
    intersection = train_sentences.intersection(val_sentences)

    if len(intersection) > 0:
        raise AssertionError(
            f"Group split failed! {len(intersection)} sentences found in both train and validation sets."
        )

    print(
        "Group split verification passed: No sentence overlap between train and validation."
    )

    print("\nMetadata generation and verification completed successfully.")


if __name__ == "__main__":
    main()
