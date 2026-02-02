import os
import struct
import sys
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Constants
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
TRAIN_BSON = os.path.join(INPUT_DIR, "train.bson")
TEST_BSON = os.path.join(INPUT_DIR, "test.bson")
RANDOM_STATE = 42

# BSON Type Constants
BSON_TYPE_DOUBLE = 1
BSON_TYPE_STRING = 2
BSON_TYPE_OBJECT = 3
BSON_TYPE_ARRAY = 4
BSON_TYPE_BINARY = 5
BSON_TYPE_OBJECTID = 7
BSON_TYPE_BOOL = 8
BSON_TYPE_DATE = 9
BSON_TYPE_NULL = 10
BSON_TYPE_INT32 = 16
BSON_TYPE_INT64 = 18


def read_c_string(buffer, start):
    """Reads a C-style null-terminated string from buffer."""
    end = buffer.find(b"\x00", start)
    if end == -1:
        return None, -1
    return buffer[start:end].decode("utf-8", errors="ignore"), end + 1


def parse_bson_document(buffer, search_keys):
    """
    Parses a BSON document buffer to find values for specific keys.
    Returns a dict of found keys and their values.
    """
    results = {}
    idx = 0
    buf_len = len(buffer)

    # Check if buffer is large enough for at least size (4 bytes)
    if buf_len < 5:
        return results

    # Skip size (already read or part of buffer, but we iterate content)
    # The buffer passed here is expected to be the payload (excluding the initial 4-byte size if handled outside,
    # or we just start parsing elements).
    # Standard BSON: [Size (4)][Element List][Null (1)]
    # We assume 'buffer' is the content AFTER the 4-byte size.

    while idx < buf_len - 1:  # -1 for the final null terminator of the document
        type_byte = buffer[idx]
        idx += 1

        # Read Name
        name, idx = read_c_string(buffer, idx)
        if idx == -1:
            break  # Incomplete buffer

        # Check if we need this key
        is_target = name in search_keys

        # Parse Value based on type
        val = None

        try:
            if type_byte == BSON_TYPE_DOUBLE:
                if idx + 8 > buf_len:
                    break
                if is_target:
                    val = struct.unpack("<d", buffer[idx : idx + 8])[0]
                idx += 8
            elif type_byte == BSON_TYPE_STRING:
                if idx + 4 > buf_len:
                    break
                s_len = struct.unpack("<i", buffer[idx : idx + 4])[0]
                if idx + 4 + s_len > buf_len:
                    break
                if is_target:
                    val = buffer[idx + 4 : idx + 4 + s_len - 1].decode(
                        "utf-8", errors="ignore"
                    )
                idx += 4 + s_len
            elif type_byte == BSON_TYPE_OBJECT or type_byte == BSON_TYPE_ARRAY:
                if idx + 4 > buf_len:
                    break
                o_len = struct.unpack("<i", buffer[idx : idx + 4])[0]
                # o_len includes the size bytes themselves
                if idx + o_len > buf_len:
                    break
                # We don't parse inside objects/arrays for metadata (unless nested keys needed, which is not the case here)
                idx += o_len
            elif type_byte == BSON_TYPE_BINARY:
                if idx + 4 > buf_len:
                    break
                b_len = struct.unpack("<i", buffer[idx : idx + 4])[0]
                if idx + 4 + 1 + b_len > buf_len:
                    break
                idx += 4 + 1 + b_len
            elif type_byte == BSON_TYPE_OBJECTID:
                if idx + 12 > buf_len:
                    break
                idx += 12
            elif type_byte == BSON_TYPE_BOOL:
                if idx + 1 > buf_len:
                    break
                idx += 1
            elif type_byte == BSON_TYPE_DATE:
                if idx + 8 > buf_len:
                    break
                idx += 8
            elif type_byte == BSON_TYPE_NULL:
                pass
            elif type_byte == BSON_TYPE_INT32:
                if idx + 4 > buf_len:
                    break
                if is_target:
                    val = struct.unpack("<i", buffer[idx : idx + 4])[0]
                idx += 4
            elif type_byte == BSON_TYPE_INT64:
                if idx + 8 > buf_len:
                    break
                if is_target:
                    val = struct.unpack("<q", buffer[idx : idx + 8])[0]
                idx += 8
            else:
                # Unknown type, cannot safely proceed
                break

            if is_target:
                results[name] = val

            # Early exit if all keys found
            if len(results) == len(search_keys):
                return results

        except struct.error:
            break

    return results


def scan_bson_file(filepath, has_category=True):
    """
    Scans a BSON file and extracts metadata.
    Returns a list of dictionaries.
    """
    metadata = []

    # We only need _id and category_id.
    # 'imgs' is usually large, so we try to skip it.
    search_keys = {"_id"}
    if has_category:
        search_keys.add("category_id")

    file_name = os.path.basename(filepath)

    # Optimization: Read a chunk to find metadata before images
    CHUNK_SIZE = 512

    with open(filepath, "rb") as f:
        while True:
            offset = f.tell()
            header_bytes = f.read(4)
            if len(header_bytes) < 4:
                break

            doc_size = struct.unpack("<i", header_bytes)[0]

            # Read a small chunk first to see if we can find keys without reading images
            # We read min(doc_size - 4, CHUNK_SIZE)
            read_len = min(doc_size - 4, CHUNK_SIZE)
            chunk = f.read(read_len)

            # Attempt parse
            results = parse_bson_document(chunk, search_keys)

            # If we didn't find all keys, we might have been unlucky (keys after images)
            # or the chunk was cut off in the middle of a field.
            # In this case, we must read the rest of the document and parse fully.
            if len(results) < len(search_keys):
                # Read remainder
                remaining_len = (doc_size - 4) - len(chunk)
                if remaining_len > 0:
                    remainder = f.read(remaining_len)
                    full_payload = chunk + remainder
                    results = parse_bson_document(full_payload, search_keys)
            else:
                # We found everything, skip the rest of the document
                remaining_len = (doc_size - 4) - len(chunk)
                if remaining_len > 0:
                    f.seek(remaining_len, 1)  # Seek forward relative

            # Store metadata
            record = {
                "sample_id": results.get("_id"),
                "bson_file_path": file_name,
                "bson_offset": offset,
                "bson_length": doc_size,
            }
            if has_category:
                record["category_id"] = results.get("category_id")

            metadata.append(record)

            # Optional: Print progress
            if len(metadata) % 100000 == 0:
                print(
                    f"Processed {len(metadata)} records from {file_name}...", flush=True
                )

    return pd.DataFrame(metadata)


def generate_metadata():
    print("Step 1: Scanning train.bson...")
    start_time = time.time()
    df_train_full = scan_bson_file(TRAIN_BSON, has_category=True)
    print(
        f"Finished scanning train.bson. Found {len(df_train_full)} records. Time: {time.time() - start_time:.2f}s"
    )

    print("Step 2: Creating Validation Split...")
    # Handle rare classes
    class_counts = df_train_full["category_id"].value_counts()
    rare_classes = class_counts[class_counts < 2].index

    # Separate rare classes (force to train)
    df_rare = df_train_full[df_train_full["category_id"].isin(rare_classes)]
    df_common = df_train_full[~df_train_full["category_id"].isin(rare_classes)]

    print(
        f"Found {len(rare_classes)} rare categories with < 2 samples. ({len(df_rare)} records)"
    )

    # Stratified split on common classes
    train_common, val_common = train_test_split(
        df_common,
        test_size=0.2,
        stratify=df_common["category_id"],
        random_state=RANDOM_STATE,
    )

    # Combine
    df_train_final = (
        pd.concat([train_common, df_rare])
        .sample(frac=1, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )
    df_val_final = val_common.sample(frac=1, random_state=RANDOM_STATE).reset_index(
        drop=True
    )

    print(f"Final Train Size: {len(df_train_final)}")
    print(f"Final Val Size: {len(df_val_final)}")

    # Save Train/Val
    os.makedirs(METADATA_DIR, exist_ok=True)
    df_train_final.to_csv(os.path.join(METADATA_DIR, "train_metadata.csv"), index=False)
    df_val_final.to_csv(os.path.join(METADATA_DIR, "val_metadata.csv"), index=False)

    print("Step 3: Scanning test.bson...")
    start_time = time.time()
    df_test = scan_bson_file(TEST_BSON, has_category=False)
    print(
        f"Finished scanning test.bson. Found {len(df_test)} records. Time: {time.time() - start_time:.2f}s"
    )

    df_test.to_csv(os.path.join(METADATA_DIR, "test_metadata.csv"), index=False)


def validate_submission():
    print("\n==== Validating Metadata ====")

    # Load metadata
    df_train = pd.read_csv(os.path.join(METADATA_DIR, "train_metadata.csv"))
    df_val = pd.read_csv(os.path.join(METADATA_DIR, "val_metadata.csv"))
    df_test = pd.read_csv(os.path.join(METADATA_DIR, "test_metadata.csv"))

    # 1. Summary Statistics
    print(
        f"Train Set: {len(df_train)} samples, {df_train['category_id'].nunique()} categories."
    )
    print(
        f"Val Set:   {len(df_val)} samples, {df_val['category_id'].nunique()} categories."
    )
    print(f"Test Set:  {len(df_test)} samples.")

    # 2. File Path Checks
    # Check if files exist
    missing_files = 0
    checked_paths = 0

    for df, name in [(df_train, "train"), (df_val, "val"), (df_test, "test")]:
        sample = df.sample(min(1000, len(df)), random_state=RANDOM_STATE)
        for _, row in sample.iterrows():
            checked_paths += 1
            rel_path = os.path.join(INPUT_DIR, row["bson_file_path"])
            if not os.path.exists(rel_path):
                missing_files += 1
                if missing_files <= 5:
                    print(f"Missing file: {rel_path}")
            else:
                # Verify offset validity
                try:
                    with open(rel_path, "rb") as f:
                        f.seek(row["bson_offset"])
                        size_bytes = f.read(4)
                        if len(size_bytes) == 4:
                            size = struct.unpack("<i", size_bytes)[0]
                            if size != row["bson_length"]:
                                print(
                                    f"Size mismatch at offset {row['bson_offset']}: Metadata {row['bson_length']} vs File {size}"
                                )
                                missing_files += 1  # Treat as invalid
                        else:
                            missing_files += 1
                except Exception as e:
                    print(f"Error checking file {rel_path}: {e}")
                    missing_files += 1

    ratio = missing_files / checked_paths if checked_paths > 0 else 0
    print(f"Missing/Invalid File Ratio: {ratio:.4f}")
    if ratio > 0.5:
        raise FileNotFoundError("More than 50% of file paths or offsets are invalid.")

    # 3. Validation Split Checks
    total_train_val = len(df_train) + len(df_val)
    val_ratio = len(df_val) / total_train_val
    print(f"Validation Ratio: {val_ratio:.4f}")

    # Assert ratio is close to 0.2 (allow small deviation due to rare class handling)
    assert 0.19 < val_ratio < 0.21, "Validation split ratio is not approximately 0.2"

    # Assert stratification
    # Compare distribution of top 10 classes
    top_classes = df_train["category_id"].value_counts().head(10).index
    train_dist = df_train["category_id"].value_counts(normalize=True)
    val_dist = df_val["category_id"].value_counts(normalize=True)

    print("Checking stratification for top 5 classes:")
    for cls in top_classes[:5]:
        t_p = train_dist.get(cls, 0)
        v_p = val_dist.get(cls, 0)
        print(f"Class {cls}: Train {t_p:.4f}, Val {v_p:.4f}")
        # Loose check because small classes might vary, but top classes should be close
        if abs(t_p - v_p) > 0.05:
            raise AssertionError(f"Stratification failed for class {cls}")

    print("Validation checks passed successfully.")


if __name__ == "__main__":
    generate_metadata()
    validate_submission()
