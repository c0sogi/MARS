import os
import struct
import pandas as pd
import numpy as np
from library.config import (
    CATEGORY_NAMES_PATH,
    CACHE_DIR,
    TRAIN_BSON_PATH,
    TEST_BSON_PATH,
    TRAIN_EXAMPLE_BSON_PATH,
)


class BSONReader:
    """
    A utility class to read and parse BSON files for image extraction.
    Designed to be thread-safe for use in DataLoaders by opening files on read.
    """

    def __init__(self, bson_path):
        self.bson_path = bson_path

    def read_images(self, offset, length):
        """
        Reads a BSON record at the specified offset and extracts image binaries.

        Args:
            offset (int): Byte offset of the record in the BSON file.
            length (int): Total length of the record in bytes.

        Returns:
            list[bytes]: A list of byte strings, each representing a raw image (e.g. JPEG).
        """
        if not os.path.exists(self.bson_path):
            raise FileNotFoundError(f"BSON file not found at {self.bson_path}")

        with open(self.bson_path, "rb") as f:
            f.seek(offset)
            data = f.read(length)

        return self._extract_images_from_bytes(data)

    def _get_val_size(self, type_byte, data, ptr):
        """Helper to determine the size of a BSON value based on its type."""
        if type_byte == 0x01:  # double
            return 8
        elif type_byte == 0x02:  # string
            s_len = struct.unpack("<i", data[ptr : ptr + 4])[0]
            return 4 + s_len
        elif type_byte == 0x03:  # document
            d_len = struct.unpack("<i", data[ptr : ptr + 4])[0]
            return d_len
        elif type_byte == 0x04:  # array
            a_len = struct.unpack("<i", data[ptr : ptr + 4])[0]
            return a_len
        elif type_byte == 0x05:  # binary
            b_len = struct.unpack("<i", data[ptr : ptr + 4])[0]
            return 4 + 1 + b_len
        elif type_byte == 0x07:  # objectid
            return 12
        elif type_byte == 0x08:  # boolean
            return 1
        elif type_byte == 0x09:  # utc datetime
            return 8
        elif type_byte == 0x0A:  # null
            return 0
        elif type_byte == 0x10:  # int32
            return 4
        elif type_byte == 0x12:  # int64
            return 8
        else:
            return 0

    def _extract_images_from_bytes(self, data):
        """
        Parses raw BSON bytes to find the 'imgs' array and extract 'picture' binaries.
        """
        images = []
        ptr = 4  # Skip total size header (int32)
        length = len(data)

        while ptr < length - 1:
            type_byte = data[ptr]
            ptr += 1

            # Read Field Name (null-terminated string)
            name_end = data.find(b"\x00", ptr)
            if name_end == -1:
                break
            name = data[ptr:name_end].decode("utf-8", errors="ignore")
            ptr = name_end + 1

            if name == "imgs" and type_byte == 0x04:
                # Found 'imgs' array
                arr_len = struct.unpack("<i", data[ptr : ptr + 4])[0]
                arr_end = ptr + arr_len

                # Enter Array (skip length int)
                ap = ptr + 4
                while ap < arr_end - 1:
                    etype = data[ap]
                    ap += 1

                    # Array keys are "0", "1"... skip them
                    ename_end = data.find(b"\x00", ap)
                    ap = ename_end + 1

                    if etype == 0x03:  # Document (Image container)
                        doc_len = struct.unpack("<i", data[ap : ap + 4])[0]
                        doc_end = ap + doc_len

                        # Enter Document
                        dp = ap + 4
                        while dp < doc_end - 1:
                            dtype = data[dp]
                            dp += 1

                            dname_end = data.find(b"\x00", dp)
                            dname = data[dp:dname_end].decode("utf-8", errors="ignore")
                            dp = dname_end + 1

                            if dname == "picture" and dtype == 0x05:
                                # Found picture binary
                                bin_len = struct.unpack("<i", data[dp : dp + 4])[0]
                                # subtype is at dp+4, data starts at dp+5
                                img_bytes = data[dp + 5 : dp + 5 + bin_len]
                                images.append(img_bytes)
                                dp += 4 + 1 + bin_len
                            else:
                                # Skip other fields in image doc
                                v_len = self._get_val_size(dtype, data, dp)
                                dp += v_len

                        ap += doc_len
                    else:
                        v_len = self._get_val_size(etype, data, ap)
                        ap += v_len

                ptr += arr_len
            else:
                # Skip this field
                v_len = self._get_val_size(type_byte, data, ptr)
                ptr += v_len

        return images


class HierarchyManager:
    """
    Manages the mapping between raw category IDs and hierarchical indices (L1, L2, L3).
    Caches the mapping to disk (Parquet) to ensure consistency and speed.
    """

    def __init__(self, load_cached_data=True):
        self.cache_path = os.path.join(CACHE_DIR, "hierarchy_map.parquet")
        self.mapping_df = self._load_or_build_mapping(load_cached_data)

        # Create fast lookup dictionaries
        # Map raw category_id to (l1_idx, l2_idx, l3_idx)
        self.cat_to_indices = self.mapping_df.set_index("category_id")[
            ["l1_idx", "l2_idx", "l3_idx"]
        ].to_dict("index")

        # Map l3_idx back to raw category_id (for submission)
        self.l3_to_cat = self.mapping_df.set_index("l3_idx")["category_id"].to_dict()

    def _load_or_build_mapping(self, load_cached_data):
        """
        Loads the hierarchy mapping from cache if available, otherwise builds it
        from category_names.csv and saves it.
        """
        if load_cached_data and os.path.exists(self.cache_path):
            return pd.read_parquet(self.cache_path)

        # Build from source
        df = pd.read_csv(CATEGORY_NAMES_PATH)

        # Encode Level 1 (Coarse)
        l1_uniques = sorted(df["category_level1"].unique())
        l1_map = {name: i for i, name in enumerate(l1_uniques)}
        df["l1_idx"] = df["category_level1"].map(l1_map)

        # Encode Level 2 (Intermediate)
        l2_uniques = sorted(df["category_level2"].unique())
        l2_map = {name: i for i, name in enumerate(l2_uniques)}
        df["l2_idx"] = df["category_level2"].map(l2_map)

        # Encode Level 3 (Fine-grained Target)
        # We sort by category_id to ensure deterministic mapping 0..N-1
        cat_ids = sorted(df["category_id"].unique())
        l3_map = {cid: i for i, cid in enumerate(cat_ids)}
        df["l3_idx"] = df["category_id"].map(l3_map)

        # Ensure cache directory exists and save
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        df.to_parquet(self.cache_path)

        return df

    def get_labels(self, category_id):
        """
        Returns the hierarchical label tuple (l1_idx, l2_idx, l3_idx) for a given raw category_id.

        Args:
            category_id (int): The raw category ID from the dataset.

        Returns:
            tuple: (l1_idx, l2_idx, l3_idx)
        """
        if category_id not in self.cat_to_indices:
            raise ValueError(f"Category ID {category_id} not found in hierarchy.")
        entry = self.cat_to_indices[category_id]
        return entry["l1_idx"], entry["l2_idx"], entry["l3_idx"]

    def get_category_id_from_l3(self, l3_idx):
        """
        Returns the raw category_id corresponding to a model's L3 prediction index.

        Args:
            l3_idx (int): The predicted class index (0..N-1).

        Returns:
            int: The original category_id.
        """
        return self.l3_to_cat.get(l3_idx, None)

    def get_num_classes(self):
        """
        Returns the number of classes for each level of the hierarchy.

        Returns:
            tuple: (num_l1, num_l2, num_l3)
        """
        return (
            int(self.mapping_df["l1_idx"].max() + 1),
            int(self.mapping_df["l2_idx"].max() + 1),
            int(self.mapping_df["l3_idx"].max() + 1),
        )
