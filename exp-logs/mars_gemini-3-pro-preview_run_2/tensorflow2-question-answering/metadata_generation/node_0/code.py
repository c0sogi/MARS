import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Constants
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
TRAIN_FILE_NAME = "simplified-nq-train.jsonl"
TEST_FILE_NAME = "simplified-nq-test.jsonl"
RANDOM_STATE = 42
VAL_SIZE = 0.2


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def parse_annotations(annotations):
    """
    Parses the annotations list to extract key label information.
    Assumes annotations is a list containing one dict (as per NQ dataset structure).
    """
    if not annotations or len(annotations) == 0:
        return -1, False, "NONE"

    ann = annotations[0]

    # Long answer
    long_answer = ann.get("long_answer", {})
    long_answer_index = long_answer.get("candidate_index", -1)

    # Short answer
    short_answers = ann.get("short_answers", [])
    has_short_answer = len(short_answers) > 0

    # Yes/No answer
    yes_no_answer = ann.get("yes_no_answer", "NONE")

    return long_answer_index, has_short_answer, yes_no_answer


def process_jsonl(file_name, is_train=True, chunksize=10000):
    file_path = os.path.join(INPUT_DIR, file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")

    print(f"Processing {file_name} in chunks...")

    meta_chunks = []

    # Read in chunks to manage memory
    reader = pd.read_json(file_path, lines=True, chunksize=chunksize)

    for i, chunk in enumerate(reader):
        # Keep only necessary columns
        if is_train:
            subset = chunk[["example_id", "annotations"]].copy()

            # Parse annotations to create columns for stratification and labels
            parsed_data = subset["annotations"].apply(parse_annotations)

            # Unzip the parsed tuple into separate columns
            # Note: zip(*series) creates tuples of columns
            long_idx, has_short, yes_no = zip(*parsed_data)

            subset["long_answer_index"] = long_idx
            subset["has_short_answer"] = has_short
            subset["yes_no_answer"] = yes_no

            # Create a composite label for stratification
            # Categories: Long(T/F)_Short(T/F)_YesNo(YES/NO/NONE)
            subset["stratify_label"] = (
                subset["long_answer_index"].apply(lambda x: "L1" if x != -1 else "L0")
                + "_"
                + subset["has_short_answer"].apply(lambda x: "S1" if x else "S0")
                + "_"
                + subset["yes_no_answer"]
            )

            # Drop raw annotations to save memory in metadata
            subset.drop(columns=["annotations"], inplace=True)

        else:
            subset = chunk[["example_id"]].copy()

        # Add file path relative to ./input (just the filename as requested by logic)
        subset["file_path"] = file_name

        meta_chunks.append(subset)

        if i % 10 == 0:
            print(f"  Processed chunk {i}...")

    print(f"Finished processing {file_name}.")
    return pd.concat(meta_chunks, ignore_index=True)


def verify_metadata(df, name):
    print(f"\n--- Verifying {name} Metadata ---")
    print(f"Shape: {df.shape}")

    # 1. Summary Statistics
    if "stratify_label" in df.columns:
        print("Class Distribution (Top 10 stratify_labels):")
        print(df["stratify_label"].value_counts(normalize=True).head(10))
        print("Yes/No Distribution:")
        print(df["yes_no_answer"].value_counts(normalize=True))

    # 2. File Path Check
    print("Checking file paths...")
    sample_size = 1000
    if len(df) > sample_size:
        sample = df.sample(n=sample_size, random_state=RANDOM_STATE)
    else:
        sample = df

    missing_count = 0
    missing_samples = []

    for _, row in sample.iterrows():
        # Path is relative to ./input
        rel_path = row["file_path"]
        full_path = os.path.join(INPUT_DIR, rel_path)

        if not os.path.exists(full_path):
            missing_count += 1
            if len(missing_samples) < 5:
                missing_samples.append(rel_path)

    missing_ratio = missing_count / len(sample)
    print(f"Missing file ratio: {missing_ratio:.4f}")

    if missing_ratio > 0.5:
        print("Sample missing paths:", missing_samples)
        raise FileNotFoundError(
            f"Too many file paths in {name} metadata do not resolve. Ratio: {missing_ratio}"
        )


def main():
    ensure_dir(METADATA_DIR)

    # --- 1. Process Training Data ---
    print("Loading training data...")
    full_train_df = process_jsonl(TRAIN_FILE_NAME, is_train=True)

    # --- 2. Split Train/Validation ---
    print("Splitting training data into Train/Validation...")
    # We use the 'stratify_label' created earlier
    train_df, val_df = train_test_split(
        full_train_df,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=full_train_df["stratify_label"],
    )

    # Save metadata
    train_csv_path = os.path.join(METADATA_DIR, "train_metadata.csv")
    val_csv_path = os.path.join(METADATA_DIR, "validation_metadata.csv")

    train_df.to_csv(train_csv_path, index=False)
    val_df.to_csv(val_csv_path, index=False)
    print(f"Saved {train_csv_path}")
    print(f"Saved {val_csv_path}")

    # --- 3. Process Test Data ---
    print("Loading test data...")
    test_df = process_jsonl(TEST_FILE_NAME, is_train=False)

    test_csv_path = os.path.join(METADATA_DIR, "test_metadata.csv")
    test_df.to_csv(test_csv_path, index=False)
    print(f"Saved {test_csv_path}")

    # --- 4. Verification ---

    # Load back to ensure integrity
    loaded_train = pd.read_csv(train_csv_path)
    loaded_val = pd.read_csv(val_csv_path)
    loaded_test = pd.read_csv(test_csv_path)

    # Verify Train
    verify_metadata(loaded_train, "Train")

    # Verify Validation
    verify_metadata(loaded_val, "Validation")

    # Verify Test
    verify_metadata(loaded_test, "Test")

    # Verify Stratification
    print("\n--- Verifying Stratification ---")
    train_dist = loaded_train["stratify_label"].value_counts(normalize=True)
    val_dist = loaded_val["stratify_label"].value_counts(normalize=True)

    # Calculate difference in distribution
    # We align the indices (labels) and fill NaN with 0
    diff = (train_dist - val_dist).abs().sum()
    print(
        f"Sum of absolute differences in class probabilities between Train and Val: {diff:.4f}"
    )

    # Assert that the distribution is reasonably similar (tolerance 0.01 for sum of diffs)
    # Since we used stratified split on a large dataset, it should be very close.
    if diff > 0.05:
        raise AssertionError(
            "Stratification failed: Validation distribution differs significantly from Train distribution."
        )

    print("\nMetadata generation and verification completed successfully.")


if __name__ == "__main__":
    main()
