import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split


def generate_metadata():
    # Define paths
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Create metadata directory
    os.makedirs(METADATA_DIR, exist_ok=True)

    # Input file paths
    train_file = os.path.join(INPUT_DIR, "en_train.csv")
    test_file = os.path.join(INPUT_DIR, "en_test_2.csv")

    print(f"Reading raw data from {INPUT_DIR}...")

    # Load Training Data
    # Using low_memory=False to prevent mixed type warnings if any, though dataset seems clean
    df_train_full = pd.read_csv(train_file)
    print(f"Loaded train data with {len(df_train_full)} rows.")

    # Ensure 'id' column exists in train (useful for tracking)
    # The description says id is sentence_id + "_" + token_id
    if "id" not in df_train_full.columns:
        df_train_full["id"] = (
            df_train_full["sentence_id"].astype(str)
            + "_"
            + df_train_full["token_id"].astype(str)
        )

    # Perform Group Split by sentence_id
    # We must not split a sentence across train and val
    print("Performing group split by sentence_id (80/20)...")
    unique_sentences = df_train_full["sentence_id"].unique()

    train_sents, val_sents = train_test_split(
        unique_sentences, test_size=0.2, random_state=42, shuffle=True
    )

    # Convert to sets for O(1) lookup
    train_sents_set = set(train_sents)
    val_sents_set = set(val_sents)

    # Filter dataframes
    df_train = df_train_full[df_train_full["sentence_id"].isin(train_sents_set)].copy()
    df_val = df_train_full[df_train_full["sentence_id"].isin(val_sents_set)].copy()

    print(f"Train split size: {len(df_train)} tokens")
    print(f"Val split size: {len(df_val)} tokens")

    # Save Train/Val Metadata
    # We save the actual tabular data as metadata for this NLP task
    train_meta_path = os.path.join(METADATA_DIR, "train.csv")
    val_meta_path = os.path.join(METADATA_DIR, "val.csv")

    df_train.to_csv(train_meta_path, index=False)
    df_val.to_csv(val_meta_path, index=False)
    print(f"Saved train metadata to {train_meta_path}")
    print(f"Saved val metadata to {val_meta_path}")

    # Load and Process Test Data
    df_test = pd.read_csv(test_file)
    print(f"Loaded test data with {len(df_test)} rows.")

    # Ensure 'id' column exists in test (required for submission)
    if "id" not in df_test.columns:
        df_test["id"] = (
            df_test["sentence_id"].astype(str) + "_" + df_test["token_id"].astype(str)
        )

    test_meta_path = os.path.join(METADATA_DIR, "test.csv")
    df_test.to_csv(test_meta_path, index=False)
    print(f"Saved test metadata to {test_meta_path}")


def validate_metadata():
    METADATA_DIR = "./metadata"
    print("\n" + "=" * 30)
    print("Validating Generated Metadata")
    print("=" * 30)

    # Load generated files
    df_train = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
    df_val = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))
    df_test = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))

    # 1. Summary Statistics
    print("\n--- Summary Statistics ---")
    print(f"Train Rows: {len(df_train)}")
    print(f"Val Rows:   {len(df_val)}")
    print(f"Test Rows:  {len(df_test)}")

    print("\nTrain Class Distribution (Top 5):")
    print(df_train["class"].value_counts().head().to_string())

    print("\nVal Class Distribution (Top 5):")
    print(df_val["class"].value_counts().head().to_string())

    print("\nUnique Sentences:")
    n_train_sents = df_train["sentence_id"].nunique()
    n_val_sents = df_val["sentence_id"].nunique()
    print(f"Train: {n_train_sents}")
    print(f"Val:   {n_val_sents}")

    # 2. File Path Check
    # This dataset contains text, not file paths.
    # However, we implement the logic structure as requested, noting it's not applicable.
    print("\n--- File Path Check ---")
    print("Dataset consists of inline text. No relative file paths to check.")

    # 3. Validation Logic
    print("\n--- Verifying Split Requirements ---")

    # Check 1: Group Split Integrity (No leakage of sentence_id)
    train_ids = set(df_train["sentence_id"].unique())
    val_ids = set(df_val["sentence_id"].unique())

    intersection = train_ids.intersection(val_ids)
    if len(intersection) > 0:
        # Print sample of leaking ids
        print(f"Leaking sentence IDs sample: {list(intersection)[:5]}")
        raise AssertionError(
            f"Split failed: {len(intersection)} sentences found in both train and validation sets."
        )
    print("PASS: No sentence leakage detected.")

    # Check 2: Split Ratio
    total_sents = n_train_sents + n_val_sents
    val_ratio = n_val_sents / total_sents
    print(f"Validation Ratio (by sentence count): {val_ratio:.4f}")

    # We expect close to 0.2
    if not (0.19 <= val_ratio <= 0.21):
        raise AssertionError(
            f"Validation split ratio {val_ratio:.4f} deviates significantly from 0.2"
        )
    print("PASS: Split ratio is within acceptable bounds.")

    print("\nAll validation checks passed successfully.")


if __name__ == "__main__":
    generate_metadata()
    validate_metadata()
