import struct
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from library.config import (
    TRAIN_BSON,
    TEST_BSON,
    METADATA_DIR,
    TRAIN_META,
    VAL_META,
    TEST_META,
    SEED,
    INPUT_DIR,
)


def parse_bson_metadata(file_path, has_category=True, limit=None):
    """
    Scans a BSON file and extracts _id, category_id (if present), offset, and length.
    Optimized to skip parsing the full document once fields are found.
    """
    rows = []
    file_name = os.path.relpath(file_path, INPUT_DIR)

    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"BSON file not found: {file_path}")

    with open(file_path, "rb") as f:
        offset = 0
        count = 0

        while True:
            if limit is not None and count >= limit:
                break

            # Read size (4 bytes)
            size_bytes = f.read(4)
            if len(size_bytes) < 4:
                break

            total_size = struct.unpack("<i", size_bytes)[0]

            # Read the rest of the document
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
            count += 1

    return pd.DataFrame(rows)


def generate_metadata(load_cached_data=True, debug=False, val_size=0.2):
    """
    Generates or loads metadata CSVs for train, validation, and test sets.

    Args:
        load_cached_data (bool): If True, attempts to load existing CSVs from METADATA_DIR.
        debug (bool): If True, processes only a small subset of data.
        val_size (float): Proportion of training data to use for validation.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    # 1. Check Cache
    if load_cached_data and not debug:
        if (
            os.path.exists(TRAIN_META)
            and os.path.exists(VAL_META)
            and os.path.exists(TEST_META)
        ):
            print("Loading metadata from cache...")
            train_df = pd.read_csv(TRAIN_META)
            val_df = pd.read_csv(VAL_META)
            test_df = pd.read_csv(TEST_META)
            return train_df, val_df, test_df

    print("Generating metadata from BSON files...")
    os.makedirs(METADATA_DIR, exist_ok=True)

    limit = 10000 if debug else None

    # 2. Process Train BSON
    print(f"Scanning {TRAIN_BSON}...")
    train_df_full = parse_bson_metadata(TRAIN_BSON, has_category=True, limit=limit)

    # Filter missing categories
    train_df_full = train_df_full.dropna(subset=["category_id"])
    train_df_full["category_id"] = train_df_full["category_id"].astype(int)

    # 3. Stratified Split
    print("Splitting train/val...")
    if debug:
        # Simple split for debug
        X_train, X_val = train_test_split(
            train_df_full, test_size=val_size, random_state=SEED
        )
    else:
        # Handle rare classes
        class_counts = train_df_full["category_id"].value_counts()
        valid_classes = class_counts[class_counts >= 2].index

        train_valid = train_df_full[train_df_full["category_id"].isin(valid_classes)]
        train_rare = train_df_full[~train_df_full["category_id"].isin(valid_classes)]

        X = train_valid
        y = train_valid["category_id"]

        X_train, X_val = train_test_split(
            X, test_size=val_size, stratify=y, random_state=SEED
        )

        # Add rare classes to train to avoid losing them
        X_train = pd.concat([X_train, train_rare])

        # Shuffle
        X_train = X_train.sample(frac=1, random_state=SEED).reset_index(drop=True)
        X_val = X_val.reset_index(drop=True)

    # 4. Process Test BSON
    print(f"Scanning {TEST_BSON}...")
    test_df = parse_bson_metadata(TEST_BSON, has_category=False, limit=limit)

    # 5. Save Metadata
    print("Saving metadata to disk...")
    X_train.to_csv(TRAIN_META, index=False)
    X_val.to_csv(VAL_META, index=False)
    test_df.to_csv(TEST_META, index=False)

    print("Metadata generation complete.")
    return X_train, X_val, test_df
