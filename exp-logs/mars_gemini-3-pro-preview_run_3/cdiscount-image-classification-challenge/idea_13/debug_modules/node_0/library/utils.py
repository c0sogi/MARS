import os
import struct
import pandas as pd
import numpy as np
import cv2
from library.config import Config


class LabelEncoder:
    """
    A simple label encoder that maps unique values to zero-indexed integers.
    Ensures deterministic mapping by sorting unique values.
    """

    def __init__(self):
        self.classes_ = None
        self.mapper_ = {}
        self.inverse_mapper_ = {}

    def fit(self, y):
        """
        Fit label encoder.
        Args:
            y: array-like of shape (n_samples,)
        """
        self.classes_ = np.unique(y)
        # Sort to ensure determinism
        self.classes_.sort()
        self.mapper_ = {label: i for i, label in enumerate(self.classes_)}
        self.inverse_mapper_ = {i: label for i, label in enumerate(self.classes_)}
        return self

    def transform(self, y):
        """
        Transform labels to normalized encoding.
        Args:
            y: array-like of shape (n_samples,)
        Returns:
            np.array of shape (n_samples,)
        """
        return np.array([self.mapper_.get(x, -1) for x in y])

    def inverse_transform(self, y):
        """
        Transform labels back to original encoding.
        Args:
            y: array-like of shape (n_samples,)
        Returns:
            np.array of shape (n_samples,)
        """
        return np.array([self.inverse_mapper_.get(x) for x in y])

    def __len__(self):
        return len(self.classes_) if self.classes_ is not None else 0


class HierarchyMap:
    """
    Manages the hierarchical relationship between Category ID (Level 3)
    and its parent categories (Level 2 and Level 1).
    Handles encoding of all levels to integer targets.
    """

    def __init__(self, load_cached_data=True):
        self.l1_encoder = LabelEncoder()
        self.l2_encoder = LabelEncoder()
        self.l3_encoder = LabelEncoder()

        # Mapping DataFrame: index is raw category_id, columns are l1_idx, l2_idx, l3_idx
        self.mapping_df = self._load_or_create_mapping(load_cached_data)

        # Fast lookup dictionaries
        self.cat_to_l1 = self.mapping_df["l1_idx"].to_dict()
        self.cat_to_l2 = self.mapping_df["l2_idx"].to_dict()
        self.cat_to_l3 = self.mapping_df["l3_idx"].to_dict()

        # Reverse lookup for L3 (Index -> Raw ID)
        self.l3_to_cat = {v: k for k, v in self.cat_to_l3.items()}

    def _load_or_create_mapping(self, load_cached_data):
        cache_path = Config.HIERARCHY_MAPPING

        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached hierarchy mapping from {cache_path}")
            df = pd.read_parquet(cache_path)

            # Re-fit encoders to ensure state consistency
            # We assume the cache contains the columns 'category_level1', 'category_level2', 'category_id'
            # to reconstruct the encoders, or we store the mapping directly.
            # Strategy: Load the original names to refit encoders, then verify consistency.
            # To simplify, we just reload the encoders from the category_names.csv logic
            # but use the cached dataframe for the final mapping if needed.
            # Actually, it's safer to just re-process from CSV if we need the Encoders to be live objects.
            # But the requirement is to use the cache.
            # Let's reconstruct encoders from the cached mapping if possible,
            # or just re-run the logic since it's fast (5k rows).
            # Given the strict requirement to use cache if available:
            pass

        # Since the category_names.csv is small (5k rows), processing it is very fast.
        # The caching is more critical for the result of the processing if it were expensive.
        # However, to strictly follow the "load_cached_data" pattern:

        if load_cached_data and os.path.exists(cache_path):
            try:
                mapping_df = pd.read_parquet(cache_path)
                # We need to populate the encoders for external use
                # We can infer classes from the mapping if we saved the original names
                # or we can just re-read the CSV to fit encoders.
                # Let's re-read CSV to fit encoders to ensure they have the string labels,
                # then use the mapping.
                cats = pd.read_csv(Config.CATEGORY_NAMES)
                self.l1_encoder.fit(cats["category_level1"])
                self.l2_encoder.fit(cats["category_level2"])
                self.l3_encoder.fit(cats["category_id"])
                return mapping_df
            except Exception as e:
                print(f"Failed to load cache: {e}. Recomputing...")

        # Compute from scratch
        print("Computing hierarchy mapping from source...")
        cats = pd.read_csv(Config.CATEGORY_NAMES)

        # Fit encoders
        self.l1_encoder.fit(cats["category_level1"])
        self.l2_encoder.fit(cats["category_level2"])
        self.l3_encoder.fit(cats["category_id"])

        # Create mapping
        mapping_df = pd.DataFrame(index=cats["category_id"])
        mapping_df["l1_idx"] = self.l1_encoder.transform(cats["category_level1"])
        mapping_df["l2_idx"] = self.l2_encoder.transform(cats["category_level2"])
        mapping_df["l3_idx"] = self.l3_encoder.transform(cats["category_id"])

        # Save cache
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        mapping_df.to_parquet(cache_path)

        return mapping_df

    def get_targets(self, category_id):
        """
        Returns the integer targets (l1, l2, l3) for a given raw category_id.
        """
        if category_id not in self.cat_to_l3:
            return -1, -1, -1
        return (
            self.cat_to_l1[category_id],
            self.cat_to_l2[category_id],
            self.cat_to_l3[category_id],
        )

    def get_l3_count(self):
        return len(self.l3_encoder)

    def get_l2_count(self):
        return len(self.l2_encoder)

    def get_l1_count(self):
        return len(self.l1_encoder)


class BSONIterator:
    """
    Iterator that reads images from a BSON file based on a metadata DataFrame.
    Supports random access via byte offsets.
    """

    def __init__(self, bson_path, metadata_df, transform=None):
        """
        Args:
            bson_path: Path to the .bson file.
            metadata_df: DataFrame containing '_id', 'bson_offset', 'bson_length'.
            transform: Optional callable to transform the images (e.g. resize).
        """
        self.bson_path = bson_path
        self.metadata = metadata_df
        self.transform = transform
        self.file_handle = None

    def _open_file(self):
        if self.file_handle is None:
            self.file_handle = open(self.bson_path, "rb")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        self._open_file()

        row = self.metadata.iloc[idx]
        offset = row["bson_offset"]
        length = row["bson_length"]

        # Seek and read
        self.file_handle.seek(offset)
        data = self.file_handle.read(length)

        # Parse images
        images_bytes = self._extract_images_from_bson(data)

        # Decode images
        images = []
        for img_bytes in images_bytes:
            # Decode from binary buffer
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is not None:
                # Convert BGR to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if self.transform:
                    img = self.transform(img)
                images.append(img)

        # If no images found (rare edge case), return empty list or handle upstream
        # The dataset guarantees 1-4 images, so images list should not be empty.

        product_id = row["_id"]
        # Return category_id if it exists (train/val), else None (test)
        category_id = row["category_id"] if "category_id" in row else None

        return product_id, images, category_id

    def _get_val_size(self, type_byte, data, ptr):
        """Helper to determine BSON value size."""
        if type_byte == 0x01:
            return 8
        elif type_byte == 0x02:
            return 4 + struct.unpack("<i", data[ptr : ptr + 4])[0]
        elif type_byte == 0x03:
            return struct.unpack("<i", data[ptr : ptr + 4])[0]
        elif type_byte == 0x04:
            return struct.unpack("<i", data[ptr : ptr + 4])[0]
        elif type_byte == 0x05:
            return 4 + 1 + struct.unpack("<i", data[ptr : ptr + 4])[0]
        elif type_byte == 0x07:
            return 12
        elif type_byte == 0x08:
            return 1
        elif type_byte == 0x09:
            return 8
        elif type_byte == 0x0A:
            return 0
        elif type_byte == 0x10:
            return 4
        elif type_byte == 0x12:
            return 8
        else:
            return 0

    def _extract_images_from_bson(self, data):
        """
        Parses BSON bytes to extract 'picture' fields from 'imgs' array.
        Optimized for the specific structure of this dataset.
        """
        images = []
        ptr = 4  # Skip total size
        length = len(data)

        while ptr < length - 1:
            type_byte = data[ptr]
            ptr += 1

            # Read field name
            name_end = data.find(b"\x00", ptr)
            name = data[ptr:name_end].decode("utf-8", errors="ignore")
            ptr = name_end + 1

            if name == "imgs" and type_byte == 0x04:
                # Inside 'imgs' array
                arr_len = struct.unpack("<i", data[ptr : ptr + 4])[0]
                arr_end = ptr + arr_len

                ap = ptr + 4
                while ap < arr_end - 1:
                    etype = data[ap]
                    ap += 1

                    # Skip array index key ("0", "1", etc)
                    ename_end = data.find(b"\x00", ap)
                    ap = ename_end + 1

                    if etype == 0x03:  # Document
                        doc_len = struct.unpack("<i", data[ap : ap + 4])[0]
                        doc_end = ap + doc_len

                        dp = ap + 4
                        while dp < doc_end - 1:
                            dtype = data[dp]
                            dp += 1

                            dname_end = data.find(b"\x00", dp)
                            dname = data[dp:dname_end].decode("utf-8", errors="ignore")
                            dp = dname_end + 1

                            if dname == "picture" and dtype == 0x05:
                                bin_len = struct.unpack("<i", data[dp : dp + 4])[0]
                                # subtype at dp+4, data at dp+5
                                img_bytes = data[dp + 5 : dp + 5 + bin_len]
                                images.append(img_bytes)
                                dp += 4 + 1 + bin_len
                            else:
                                dp += self._get_val_size(dtype, data, dp)

                        ap += doc_len
                    else:
                        ap += self._get_val_size(etype, data, ap)

                ptr += arr_len
            else:
                ptr += self._get_val_size(type_byte, data, ptr)

        return images

    def close(self):
        if self.file_handle:
            self.file_handle.close()
            self.file_handle = None

    def __del__(self):
        self.close()
