import os
import json
import pandas as pd
import glob
from sklearn.model_selection import train_test_split

# Configuration
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
TRAIN_FILENAME_PATTERN = "simplified-nq-train.jsonl"
TEST_FILENAME_PATTERN = (
    "*test.jsonl"  # Matches simplified-nq-test.jsonl or simplified-nq-kaggle-test.jsonl
)
RANDOM_STATE = 42
VAL_SIZE = 0.2


def get_file_path(pattern):
    """Finds a file in INPUT_DIR matching the pattern."""
    search_path = os.path.join(INPUT_DIR, pattern)
    files = glob.glob(search_path)
    if not files:
        raise FileNotFoundError(
            f"No file found matching pattern: {pattern} in {INPUT_DIR}"
        )
    # Prefer exact match if available, otherwise take the first one
    return files[0]


def parse_jsonl_metadata(file_path, is_train=True):
    """
    Reads a JSONL file and extracts metadata including byte offsets.
    Returns a list of dictionaries.
    """
    metadata = []
    filename = os.path.basename(file_path)

    print(f"Processing {filename}...")

    with open(file_path, "rb") as f:
        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                break

            try:
                # Decode line to string
                line_str = line.decode("utf-8")
                data = json.loads(line_str)

                record = {
                    "example_id": data["example_id"],
                    "file_path": filename,
                    "byte_offset": offset,
                }

                if is_train:
                    # Extract label information for stratification and ground truth
                    annotations = data.get("annotations", [])
                    has_long_answer = False
                    has_short_answer = False

                    for ann in annotations:
                        # Check for long answer
                        if ann["long_answer"]["start_token"] != -1:
                            has_long_answer = True

                        # Check for short answer (list of spans or yes/no)
                        if (
                            len(ann["short_answers"]) > 0
                            or ann["yes_no_answer"] != "NONE"
                        ):
                            has_short_answer = True

                    record["has_long_answer"] = has_long_answer
                    record["has_short_answer"] = has_short_answer

                    # Define stratification label
                    # Hierarchy: Short Answer (implies Long usually) > Long Answer Only > None
                    if has_short_answer:
                        record["stratify_label"] = "short"
                    elif has_long_answer:
                        record["stratify_label"] = "long"
                    else:
                        record["stratify_label"] = "none"

                metadata.append(record)

            except json.JSONDecodeError:
                print(f"Warning: Failed to decode JSON at offset {offset}")
                continue
            except KeyError as e:
                print(f"Warning: Missing key {e} at offset {offset}")
                continue

    return pd.DataFrame(metadata)


def validate_metadata(train_df, val_df, test_df):
    """
    Performs validation checks on the generated metadata.
    """
    print("\n--- Validation & Statistics ---")

    # 1. Summary Statistics
    print(f"Training Set Size: {len(train_df)}")
    print(f"Validation Set Size: {len(val_df)}")
    print(f"Test Set Size: {len(test_df)}")

    if "stratify_label" in train_df.columns:
        print("\nTraining Label Distribution:")
        print(train_df["stratify_label"].value_counts(normalize=True))
        print("\nValidation Label Distribution:")
        print(val_df["stratify_label"].value_counts(normalize=True))

    # 2. File Path Verification
    print("\nVerifying file paths...")
    datasets = [("Train", train_df), ("Validation", val_df), ("Test", test_df)]

    for name, df in datasets:
        if df.empty:
            continue

        # Randomly sample up to 1000 paths
        sample_size = min(1000, len(df))
        sample = df.sample(n=sample_size, random_state=RANDOM_STATE)

        missing_count = 0
        missing_examples = []

        for _, row in sample.iterrows():
            full_path = os.path.join(INPUT_DIR, row["file_path"])
            if not os.path.exists(full_path):
                missing_count += 1
                if len(missing_examples) < 5:
                    missing_examples.append(row["file_path"])

        missing_ratio = missing_count / sample_size
        print(f"{name}: Missing file ratio = {missing_ratio:.4f}")

        if missing_ratio > 0.5:
            print(f"Examples of missing paths in {name}: {missing_examples}")
            raise FileNotFoundError(
                f"Error: More than 50% of file paths in {name} metadata are invalid."
            )

    # 3. Stratification Verification
    print("\nVerifying stratification...")
    train_dist = train_df["stratify_label"].value_counts(normalize=True)
    val_dist = val_df["stratify_label"].value_counts(normalize=True)

    tolerance = 0.015  # Allow 1.5% deviation
    for label in train_dist.index:
        train_prop = train_dist[label]
        val_prop = val_dist.get(label, 0)
        diff = abs(train_prop - val_prop)

        if diff > tolerance:
            raise AssertionError(
                f"Stratification failed for label '{label}'. "
                f"Train prop: {train_prop:.4f}, Val prop: {val_prop:.4f}, Diff: {diff:.4f} > {tolerance}"
            )

    print("Stratification check passed.")


def main():
    # Ensure metadata directory exists
    os.makedirs(METADATA_DIR, exist_ok=True)

    # Locate files
    train_file_path = get_file_path(TRAIN_FILENAME_PATTERN)
    test_file_path = get_file_path(TEST_FILENAME_PATTERN)

    # 1. Process Training Data
    df_full_train = parse_jsonl_metadata(train_file_path, is_train=True)

    # 2. Split Training Data
    print("Splitting training data into Train and Validation sets...")
    train_df, val_df = train_test_split(
        df_full_train,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=df_full_train["stratify_label"],
    )

    # 3. Process Test Data
    df_test = parse_jsonl_metadata(test_file_path, is_train=False)

    # 4. Save Metadata
    print("Saving metadata files...")
    train_df.to_csv(os.path.join(METADATA_DIR, "train_metadata.csv"), index=False)
    val_df.to_csv(os.path.join(METADATA_DIR, "val_metadata.csv"), index=False)
    df_test.to_csv(os.path.join(METADATA_DIR, "test_metadata.csv"), index=False)

    # 5. Validate
    validate_metadata(train_df, val_df, df_test)

    print("\nMetadata generation completed successfully.")


if __name__ == "__main__":
    main()
