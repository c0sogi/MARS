import os
import struct
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split

# Configuration
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
TRAIN_FILE = "train.bson"
TEST_FILE = "test.bson"
RANDOM_STATE = 42


def get_bson_metadata(filename, has_label=True):
    filepath = os.path.join(INPUT_DIR, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"{filepath} not found.")

    rows = []

    # BSON types
    TYPE_DOUBLE = 1
    TYPE_STRING = 2
    TYPE_DOC = 3
    TYPE_ARRAY = 4
    TYPE_BINARY = 5
    TYPE_BOOL = 8
    TYPE_INT32 = 16
    TYPE_INT64 = 18

    # Other types to handle safely if encountered
    TYPE_OBJECT_ID = 7
    TYPE_DATETIME = 9
    TYPE_NULL = 10

    print(f"Processing {filename}...")
    start_time = time.time()
    count = 0

    with open(filepath, "rb") as f:
        while True:
            offset = f.tell()
            size_bytes = f.read(4)
            if len(size_bytes) < 4:
                break

            obj_size = struct.unpack("<i", size_bytes)[0]

            # Read the rest of the object
            # We read the full object to ensure we can parse it safely.
            # Given the large RAM, this is acceptable.
            data = f.read(obj_size - 4)
            if len(data) < obj_size - 4:
                break

            # Parse
            p = 0
            _id = None
            category_id = None

            # Use memoryview to avoid copying bytes for slicing
            mv = memoryview(data)
            limit = len(data) - 1  # last byte is 0x00

            while p < limit:
                dtype = data[p]
                p += 1

                # Read name (cstring)
                name_start = p
                while p < limit and data[p] != 0:
                    p += 1

                # Decode name
                name = data[name_start:p].decode("utf-8", errors="ignore")
                p += 1  # skip null

                # Parse value
                val = None

                if dtype == TYPE_DOUBLE:
                    p += 8
                elif dtype == TYPE_STRING:
                    l = struct.unpack_from("<i", mv, p)[0]
                    p += 4 + l
                elif dtype == TYPE_DOC:
                    l = struct.unpack_from("<i", mv, p)[0]
                    p += l
                elif dtype == TYPE_ARRAY:
                    l = struct.unpack_from("<i", mv, p)[0]
                    p += l
                elif dtype == TYPE_BINARY:
                    l = struct.unpack_from("<i", mv, p)[0]
                    p += 4 + 1 + l
                elif dtype == TYPE_BOOL:
                    p += 1
                elif dtype == TYPE_INT32:
                    val = struct.unpack_from("<i", mv, p)[0]
                    p += 4
                elif dtype == TYPE_INT64:
                    val = struct.unpack_from("<q", mv, p)[0]
                    p += 8
                elif dtype == TYPE_NULL:
                    pass
                elif dtype == TYPE_OBJECT_ID:
                    p += 12
                elif dtype == TYPE_DATETIME:
                    p += 8
                else:
                    # If we hit an unknown type, we might lose sync.
                    # Attempt to skip a reasonable amount or raise error.
                    # For this dataset, we should be fine.
                    raise ValueError(
                        f"Unknown BSON type {dtype} at offset {offset}+{p} in {filename}"
                    )

                if name == "_id":
                    _id = val
                elif name == "category_id":
                    category_id = val

                # Optimization: if we have what we need, can we break?
                # We can break the parsing loop for this object,
                # but we are already ready for the next object since we read by obj_size.
                if has_label:
                    if _id is not None and category_id is not None:
                        break
                else:
                    if _id is not None:
                        break

            row = {
                "product_id": _id,
                "category_id": category_id,
                "bson_offset": offset,
                "bson_length": obj_size,
                "file_path": filename,
            }
            rows.append(row)
            count += 1

            if count % 100000 == 0:
                elapsed = time.time() - start_time
                print(
                    f"Processed {count} records... ({count/elapsed:.2f} rec/s)",
                    end="\r",
                )

    print(f"\nFinished {filename}: {count} records.")
    return pd.DataFrame(rows)


def main():
    os.makedirs(METADATA_DIR, exist_ok=True)

    # 1. Generate Metadata for Train
    print("Generating metadata for train.bson...")
    df_train_full = get_bson_metadata(TRAIN_FILE, has_label=True)

    # 2. Generate Metadata for Test
    print("Generating metadata for test.bson...")
    df_test = get_bson_metadata(TEST_FILE, has_label=False)

    # 3. Split Train/Val
    print("Splitting train/val...")

    # Handle stratification issues with rare classes
    # Filter out classes with < 2 samples
    class_counts = df_train_full["category_id"].value_counts()
    single_sample_classes = class_counts[class_counts < 2].index

    mask_single = df_train_full["category_id"].isin(single_sample_classes)
    df_single = df_train_full[mask_single]
    df_multi = df_train_full[~mask_single]

    print(
        f"Found {len(df_single)} samples with unique categories (cannot be stratified). Adding to train."
    )

    # Stratified split
    train_split, val_split = train_test_split(
        df_multi,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df_multi["category_id"],
    )

    # Combine
    df_train = pd.concat([train_split, df_single])
    df_val = val_split

    # Shuffle train
    df_train = df_train.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)

    # Save
    print("Saving metadata files...")
    df_train.to_csv(os.path.join(METADATA_DIR, "train.csv"), index=False)
    df_val.to_csv(os.path.join(METADATA_DIR, "val.csv"), index=False)
    df_test.to_csv(os.path.join(METADATA_DIR, "test.csv"), index=False)

    # 4. Summary Statistics
    print("\n==== Summary Statistics ====")
    print(f"Train set: {len(df_train)} samples")
    print(f"Validation set: {len(df_val)} samples")
    print(f"Test set: {len(df_test)} samples")
    print(f"Train unique categories: {df_train['category_id'].nunique()}")
    print(f"Val unique categories: {df_val['category_id'].nunique()}")

    # 5. Check file paths
    print("\n==== Checking File Paths ====")
    for name, df in [("train", df_train), ("val", df_val), ("test", df_test)]:
        sample = df.sample(n=min(1000, len(df)), random_state=RANDOM_STATE)
        missing_count = 0
        for _, row in sample.iterrows():
            path = os.path.join(INPUT_DIR, row["file_path"])
            if not os.path.exists(path):
                missing_count += 1

        ratio = missing_count / len(sample)
        print(f"{name} missing file ratio: {ratio:.4f}")
        if ratio > 0.5:
            raise FileNotFoundError(f"Too many missing files in {name}")

    # 6. Verify Stratification
    print("\n==== Verifying Stratification ====")
    expected_val_cats = set(df_multi["category_id"].unique())
    actual_val_cats = set(df_val["category_id"].unique())

    # Check that most categories are present
    # Small categories (n=2,3,4) might not end up in Val due to 20% split rounding down to 0 for n<3?
    # Actually train_test_split with stratify tries to preserve distribution.
    # For n=2, 0.2*2 = 0.4 -> 0? Or 1?
    # Usually it ensures at least 1 if possible, but let's check overlap.
    overlap = len(actual_val_cats.intersection(expected_val_cats))
    print(f"Categories in Val: {len(actual_val_cats)}")
    print(
        f"Overlap with Multi-sample Train Categories: {overlap} / {len(expected_val_cats)}"
    )

    if len(actual_val_cats) < len(expected_val_cats) * 0.5:
        # This is a loose check, but if we lost 50% of categories, something is wrong.
        raise AssertionError(
            "Validation set does not represent categories well enough."
        )

    # 7. Verify BSON reading (random check)
    print("\n==== Verifying BSON Reading ====")

    def verify_offsets(df, n=5):
        sample = df.sample(n=min(n, len(df)), random_state=RANDOM_STATE)
        for _, row in sample.iterrows():
            path = os.path.join(INPUT_DIR, row["file_path"])
            offset = row["bson_offset"]
            length = row["bson_length"]
            pid = row["product_id"]

            with open(path, "rb") as f:
                f.seek(offset)
                # Read size header
                sz_bytes = f.read(4)
                if len(sz_bytes) < 4:
                    raise ValueError(f"Could not read size at offset {offset}")
                sz = struct.unpack("<i", sz_bytes)[0]

                if sz != length:
                    raise ValueError(
                        f"BSON size mismatch at offset {offset}. Expected {length}, got {sz}"
                    )

                # Read full object to ensure it's readable
                f.seek(offset)
                data = f.read(length)
                if len(data) != length:
                    raise ValueError(f"Incomplete read at offset {offset}")

    print("Checking Train samples...")
    verify_offsets(df_train)
    print("Checking Test samples...")
    verify_offsets(df_test)
    print("Verification passed successfully.")


if __name__ == "__main__":
    main()
