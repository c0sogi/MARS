import os
import struct
import pandas as pd
import numpy as np
import cv2
from library.config import Config


class HierarchyMapper:
    """
    Manages the mapping between raw category_ids and hierarchical integer indices (L1, L2, L3).
    Implements caching to parquet to speed up initialization.
    """

    def __init__(self, load_cached_data=True):
        self.map_df = None
        self.l3_to_cat_id = {}

        # Ensure working directory exists
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

        cache_path = Config.HIERARCHY_MAP_PATH

        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading hierarchy map from {cache_path}")
            self.map_df = pd.read_parquet(cache_path)
        else:
            print("Building hierarchy map from source...")
            self.map_df = self._build_mappings()
            # Save to cache
            self.map_df.to_parquet(cache_path)

        # Create reverse lookup for submission (L3 index -> Raw Category ID)
        # The map_df index is the raw category_id
        self.map_df_reset = self.map_df.reset_index()
        self.l3_to_cat_id = dict(
            zip(self.map_df_reset["l3_idx"], self.map_df_reset["category_id"])
        )

    def _build_mappings(self):
        """
        Parses category_names.csv and creates integer encodings for L1, L2, and L3.
        """
        df = pd.read_csv(Config.CATEGORY_NAMES)

        # 1. Encode Level 1 (Coarse)
        l1_uniques = sorted(df["category_level1"].unique())
        l1_map = {name: i for i, name in enumerate(l1_uniques)}
        df["l1_idx"] = df["category_level1"].map(l1_map)

        # 2. Encode Level 2 (Sub-category)
        l2_uniques = sorted(df["category_level2"].unique())
        l2_map = {name: i for i, name in enumerate(l2_uniques)}
        df["l2_idx"] = df["category_level2"].map(l2_map)

        # 3. Encode Level 3 (Fine-grained / Target)
        # Ideally, category_id maps 1-to-1 with the lowest level.
        # We map raw category_id to 0..N-1 based on sorted order of category_ids for determinism.
        cat_ids = sorted(df["category_id"].unique())
        l3_map = {cid: i for i, cid in enumerate(cat_ids)}
        df["l3_idx"] = df["category_id"].map(l3_map)

        # Set category_id as index for fast lookup
        df = df.set_index("category_id")

        # Keep only indices
        return df[["l1_idx", "l2_idx", "l3_idx"]]

    def get_labels(self, category_id):
        """
        Returns (l1_idx, l2_idx, l3_idx) for a given raw category_id.
        """
        if category_id not in self.map_df.index:
            # Should not happen given dataset integrity, but handle safely
            return -1, -1, -1

        row = self.map_df.loc[category_id]
        return int(row["l1_idx"]), int(row["l2_idx"]), int(row["l3_idx"])

    def get_category_id(self, l3_idx):
        """
        Converts a predicted L3 index back to the raw category_id.
        """
        return self.l3_to_cat_id.get(l3_idx, -1)

    def get_num_classes(self):
        """
        Returns dictionary of class counts for validation.
        """
        return {
            "l1": self.map_df["l1_idx"].max() + 1,
            "l2": self.map_df["l2_idx"].max() + 1,
            "l3": self.map_df["l3_idx"].max() + 1,
        }


class BSONIterator:
    """
    Handles reading raw BSON files and extracting images.
    """

    def __init__(self, bson_path):
        self.bson_path = bson_path
        self.file_handle = open(self.bson_path, "rb")

    def __del__(self):
        if hasattr(self, "file_handle") and self.file_handle:
            self.file_handle.close()

    def get_images(self, offset, length):
        """
        Seeks to the specific offset, reads the BSON document, extracts and decodes images.
        Returns a list of numpy arrays (images).
        """
        self.file_handle.seek(offset)
        doc_data = self.file_handle.read(length)

        img_binaries = self._extract_images_from_bson(doc_data)
        images = []

        for img_bytes in img_binaries:
            # Decode image
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                continue

            # Resize to Config dimensions
            img = cv2.resize(img, (Config.RESIZE_SIZE, Config.RESIZE_SIZE))

            # Convert BGR to RGB (standard for PyTorch pre-trained models)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            images.append(img)

        return images

    def _get_val_size(self, type_byte, data, ptr):
        """Helper to determine BSON value size."""
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

    def _extract_images_from_bson(self, data):
        """
        Parses a raw BSON document byte string to find 'imgs' array and extract 'picture' binaries.
        """
        images = []
        ptr = 4  # Skip total size header
        length = len(data)

        while ptr < length - 1:
            type_byte = data[ptr]
            ptr += 1

            # Read Field Name
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
