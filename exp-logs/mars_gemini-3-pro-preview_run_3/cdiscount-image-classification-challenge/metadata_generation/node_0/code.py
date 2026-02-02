import struct
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import time

# Configuration
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
TRAIN_BSON_PATH = os.path.join(INPUT_DIR, "train.bson")
TEST_BSON_PATH = os.path.join(INPUT_DIR, "test.bson")
RANDOM_STATE = 42
VAL_SIZE = 0.2


def parse_bson_metadata(file_path, has_category=True):
    """
    Scans a BSON file and extracts _id, category_id (if present), offset, and length.
    Optimized to skip parsing the full document once fields are found.
    """
    rows = []

    file_name = os.path.relpath(file_path, INPUT_DIR)

    with open(file_path, "rb") as f:
        offset = 0
        while True:
            # Read size (4 bytes)
            size_bytes = f.read(4)
            if len(size_bytes) < 4:
                break

            total_size = struct.unpack("<i", size_bytes)[0]

            # Read the rest of the document
            # We read the full document to memory to ensure safe parsing.
            # Given average record size is small (~10-50KB), this is efficient.
            doc_data = f.read(total_size - 4)
            if len(doc_data) < total_size - 4:
                break  # Unexpected EOF

            # Parse BSON elements
            p = 0
            _id = None
            category_id = None

            # Iterate elements
            while p < len(doc_data) - 1:  # Last byte is 0x00
                type_byte = doc_data[p]
                p += 1

                # Find null terminator for the name
                name_end = doc_data.find(b"\x00", p)
                if name_end == -1:
                    break

                name = doc_data[p:name_end].decode("utf-8", errors="ignore")
                p = name_end + 1

                # Process value based on type
                val = None

                if type_byte == 0x10:  # int32
                    if p + 4 > len(doc_data):
                        break
                    val = struct.unpack("<i", doc_data[p : p + 4])[0]
                    p += 4
                elif type_byte == 0x12:  # int64
                    if p + 8 > len(doc_data):
                        break
                    val = struct.unpack("<q", doc_data[p : p + 8])[0]
                    p += 8
                elif type_byte == 0x01:  # double
                    if p + 8 > len(doc_data):
                        break
                    val = struct.unpack("<d", doc_data[p : p + 8])[0]
                    p += 8
                    if name == "category_id":
                        val = int(val)
                elif type_byte == 0x02:  # string
                    if p + 4 > len(doc_data):
                        break
                    s_len = struct.unpack("<i", doc_data[p : p + 4])[0]
                    p += 4 + s_len
                elif type_byte == 0x03 or type_byte == 0x04:  # document or array
                    if p + 4 > len(doc_data):
                        break
                    doc_size = struct.unpack("<i", doc_data[p : p + 4])[0]
                    p += doc_size
                elif type_byte == 0x05:  # binary
                    if p + 4 > len(doc_data):
                        break
                    b_len = struct.unpack("<i", doc_data[p : p + 4])[0]
                    p += 4 + 1 + b_len  # length + subtype + bytes
                elif type_byte == 0x08:  # boolean
                    p += 1
                elif type_byte == 0x0A:  # null
                    pass
                else:
                    # Unknown type, break to avoid incorrect parsing
                    break

                if name == "_id":
                    _id = val
                elif name == "category_id":
                    category_id = val

                # Optimization: If we have what we need, stop parsing this document
                if has_category:
                    if _id is not None and category_id is not None:
                        break
                else:
                    if _id is not None:
                        break

            row = {
                "_id": _id,
                "bson_offset": offset,
                "bson_length": total_size,
                "file_path": file_name,
            }
            if has_category:
                row["category_id"] = category_id

            rows.append(row)

            offset += total_size

            if len(rows) % 200000 == 0:
                print(f"Processed {len(rows)} records from {file_name}...")

    return pd.DataFrame(rows)


def validate_paths(df, base_dir):
    """
    Checks 1000 random paths.
    """
    if len(df) == 0:
        return
    sample = df.sample(n=min(1000, len(df)), random_state=42)
    missing_count = 0
    missing_samples = []

    for _, row in sample.iterrows():
        path = os.path.join(base_dir, row["file_path"])
        if not os.path.exists(path):
            missing_count += 1
            if len(missing_samples) < 5:
                missing_samples.append(row["file_path"])

    ratio = missing_count / len(sample)
    if ratio > 0.5:
        print("Missing paths sample:", missing_samples)
        raise FileNotFoundError(f"Missing file ratio {ratio} > 0.5")
    print(
        f"Path validation passed for sample of size {len(sample)}. Missing ratio: {ratio}"
    )


if __name__ == "__main__":
    start_time = time.time()
    os.makedirs(METADATA_DIR, exist_ok=True)

    print("Scanning train.bson...")
    train_df_full = parse_bson_metadata(TRAIN_BSON_PATH, has_category=True)
    print(f"Scanned {len(train_df_full)} training records.")

    # Stratified Split
    print("Splitting train/val...")
    # Filter out records with missing category_id (should be none, but safe check)
    train_df_full = train_df_full.dropna(subset=["category_id"])
    train_df_full["category_id"] = train_df_full["category_id"].astype(int)

    # Handle classes with too few samples for stratification
    class_counts = train_df_full["category_id"].value_counts()
    valid_classes = class_counts[class_counts >= 2].index

    train_valid = train_df_full[train_df_full["category_id"].isin(valid_classes)]
    train_rare = train_df_full[~train_df_full["category_id"].isin(valid_classes)]

    print(f"Classes with < 2 samples: {len(train_rare)}")

    X = train_valid
    y = train_valid["category_id"]

    X_train, X_val = train_test_split(
        X, test_size=VAL_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    # Add rare classes to train set to avoid losing them
    X_train = pd.concat([X_train, train_rare])

    # Shuffle train set
    X_train = X_train.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    X_val = X_val.reset_index(drop=True)

    print("Saving train/val metadata...")
    X_train.to_csv(os.path.join(METADATA_DIR, "train.csv"), index=False)
    X_val.to_csv(os.path.join(METADATA_DIR, "val.csv"), index=False)

    print("Scanning test.bson...")
    test_df = parse_bson_metadata(TEST_BSON_PATH, has_category=False)
    print(f"Scanned {len(test_df)} test records.")

    print("Saving test metadata...")
    test_df.to_csv(os.path.join(METADATA_DIR, "test.csv"), index=False)

    # Validation
    print("\n=== Validation ===")

    train_meta = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
    val_meta = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))
    test_meta = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))

    print(f"Train samples: {len(train_meta)}")
    print(f"Val samples: {len(val_meta)}")
    print(f"Test samples: {len(test_meta)}")

    print("Train Class Distribution (Top 5):")
    print(train_meta["category_id"].value_counts(normalize=True).head())
    print("Val Class Distribution (Top 5):")
    print(val_meta["category_id"].value_counts(normalize=True).head())

    # Verify stratification success
    # Check if validation set size is approximately correct
    total_train_val = len(train_meta) + len(val_meta)
    actual_val_ratio = len(val_meta) / total_train_val
    print(f"Actual Validation Ratio: {actual_val_ratio:.4f} (Target: {VAL_SIZE})")

    if abs(actual_val_ratio - VAL_SIZE) > 0.05:
        raise AssertionError(
            f"Validation split ratio {actual_val_ratio} deviates significantly from {VAL_SIZE}"
        )

    # Check file paths
    print("Checking file paths...")
    validate_paths(train_meta, INPUT_DIR)
    validate_paths(val_meta, INPUT_DIR)
    validate_paths(test_meta, INPUT_DIR)

    print(f"Done in {time.time() - start_time:.2f} seconds.")
