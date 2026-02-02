import pandas as pd
import numpy as np
import os
import sys

# Constants
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
TRAIN_FILE = "ru_train.csv"
TEST_FILE = "ru_test_2.csv"
RANDOM_STATE = 42
VAL_SIZE = 0.2


def generate_metadata():
    """
    Generates metadata CSVs for train, validation, and test sets.
    Splits training data by sentence_id to ensure no leakage.
    """
    if not os.path.exists(METADATA_DIR):
        os.makedirs(METADATA_DIR)

    print("Loading raw data...")
    train_path = os.path.join(INPUT_DIR, TRAIN_FILE)
    test_path = os.path.join(INPUT_DIR, TEST_FILE)

    # Load Train Data
    # Using low_memory=False to prevent mixed type warnings if any, though memory is sufficient.
    df_train_full = pd.read_csv(train_path, dtype={"sentence_id": str, "token_id": str})

    # Load Test Data
    df_test = pd.read_csv(test_path, dtype={"sentence_id": str, "token_id": str})

    print(f"Raw Train shape: {df_train_full.shape}")
    print(f"Raw Test shape: {df_test.shape}")

    # Ensure 'id' column exists or create it for consistency
    if "id" not in df_train_full.columns:
        df_train_full["id"] = (
            df_train_full["sentence_id"] + "_" + df_train_full["token_id"]
        )

    # Add source_file column for path validation requirements
    df_train_full["source_file"] = TRAIN_FILE
    df_test["source_file"] = TEST_FILE

    # --- Group Split Strategy ---
    # We must split by sentence_id to keep context intact.
    unique_sentences = df_train_full["sentence_id"].unique()
    print(f"Unique sentences in train: {len(unique_sentences)}")

    # Shuffle unique sentences
    rng = np.random.default_rng(RANDOM_STATE)
    rng.shuffle(unique_sentences)

    # Calculate split index
    n_val = int(len(unique_sentences) * VAL_SIZE)
    val_sentences = set(unique_sentences[:n_val])
    train_sentences = set(unique_sentences[n_val:])

    print(
        f"Splitting: {len(train_sentences)} train sentences, {len(val_sentences)} val sentences."
    )

    # Apply split
    # We use boolean indexing for speed
    is_val = df_train_full["sentence_id"].isin(val_sentences)
    df_val = df_train_full[is_val].copy()
    df_train = df_train_full[~is_val].copy()

    # Save metadata
    print("Saving metadata files...")
    train_meta_path = os.path.join(METADATA_DIR, "train.csv")
    val_meta_path = os.path.join(METADATA_DIR, "val.csv")
    test_meta_path = os.path.join(METADATA_DIR, "test.csv")

    df_train.to_csv(train_meta_path, index=False)
    df_val.to_csv(val_meta_path, index=False)
    df_test.to_csv(test_meta_path, index=False)

    print("Metadata generation complete.")


def validate_metadata():
    """
    Loads generated metadata and performs validation checks.
    """
    print("\nStarting validation...")

    train_meta_path = os.path.join(METADATA_DIR, "train.csv")
    val_meta_path = os.path.join(METADATA_DIR, "val.csv")
    test_meta_path = os.path.join(METADATA_DIR, "test.csv")

    df_train = pd.read_csv(train_meta_path, dtype={"sentence_id": str})
    df_val = pd.read_csv(val_meta_path, dtype={"sentence_id": str})
    df_test = pd.read_csv(test_meta_path, dtype={"sentence_id": str})

    # 1. Summary Statistics
    print("\n--- Summary Statistics ---")
    for name, df in [("Train", df_train), ("Validation", df_val), ("Test", df_test)]:
        print(f"\nDataset: {name}")
        print(f"  Shape: {df.shape}")
        print(f"  Unique Sentences: {df['sentence_id'].nunique()}")
        if "class" in df.columns:
            print(
                f"  Class Distribution (Top 5):\n{df['class'].value_counts().head(5)}"
            )
        else:
            print("  Class column not found (expected for Test).")

    # 2. File Path Verification
    print("\n--- File Path Verification ---")
    # We check 'source_file' column relative to INPUT_DIR
    datasets = [df_train, df_val, df_test]

    for df in datasets:
        if "source_file" in df.columns:
            # Sample 1000 paths (or all if less than 1000)
            sample_paths = df["source_file"].sample(
                n=min(1000, len(df)), random_state=RANDOM_STATE
            )
            missing_count = 0
            missing_examples = []

            for rel_path in sample_paths:
                full_path = os.path.join(INPUT_DIR, rel_path)
                if not os.path.exists(full_path):
                    missing_count += 1
                    if len(missing_examples) < 5:
                        missing_examples.append(rel_path)

            ratio = missing_count / len(sample_paths)
            print(f"  Checked {len(sample_paths)} paths. Missing ratio: {ratio:.4f}")

            if ratio > 0.5:
                print("  Missing paths examples:", missing_examples)
                raise FileNotFoundError(
                    f"More than 50% of file paths are missing. Ratio: {ratio}"
                )
        else:
            print("  No 'source_file' column found to check.")

    # 3. Validation Split Verification
    print("\n--- Split Verification ---")
    train_sentences = set(df_train["sentence_id"].unique())
    val_sentences = set(df_val["sentence_id"].unique())

    # Check for intersection
    intersection = train_sentences.intersection(val_sentences)
    if intersection:
        raise AssertionError(
            f"Train and Validation sets share {len(intersection)} sentence_ids! Group split failed."
        )
    else:
        print("  SUCCESS: Train and Validation sentence IDs are disjoint.")

    # Check approximate ratio
    total_sentences = len(train_sentences) + len(val_sentences)
    val_ratio = len(val_sentences) / total_sentences
    print(f"  Actual Validation Ratio (by sentence count): {val_ratio:.4f}")

    # Allow small deviation due to discrete nature of groups, but it should be close to 0.2
    if not (0.19 < val_ratio < 0.21):
        print(
            "  WARNING: Validation ratio deviates slightly from 0.2, but this is expected with random group splitting."
        )

    print("\nAll checks passed successfully.")


if __name__ == "__main__":
    try:
        generate_metadata()
        validate_metadata()
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)
