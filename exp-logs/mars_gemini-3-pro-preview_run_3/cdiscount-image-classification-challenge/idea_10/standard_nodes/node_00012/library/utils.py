import os
import struct
import pandas as pd
import numpy as np
import cv2
from library.config import Config


class HierarchyMapper:
    """
    Handles mapping between raw category_ids and hierarchical integer targets (L1, L2, L3).
    Uses a Parquet cache to store the computed mappings.
    """

    def __init__(self, category_names_path=Config.CATEGORY_NAMES):
        self.category_names_path = category_names_path
        self.cat_id_to_hierarchy = {}  # category_id -> (l1_idx, l2_idx, l3_idx)
        self.l3_inv_map = {}  # l3_idx -> category_id

    def process(self, load_cached_data=True, cache_path=Config.HIERARCHY_MAPPING_PATH):
        """
        Loads mappings from cache or computes them from source CSV.

        Args:
            load_cached_data (bool): If True, attempts to load from cache_path.
            cache_path (str): Path to the parquet cache file.
        """
        if load_cached_data and os.path.exists(cache_path):
            df = pd.read_parquet(cache_path)
        else:
            if not os.path.exists(self.category_names_path):
                raise FileNotFoundError(
                    f"Category names file not found: {self.category_names_path}"
                )

            df = pd.read_csv(self.category_names_path)

            # Ensure category_id is int
            df["category_id"] = df["category_id"].astype(int)

            # Create encoders for each level
            # Level 1
            l1_uniques = sorted(df["category_level1"].unique())
            l1_encoder = {name: i for i, name in enumerate(l1_uniques)}

            # Level 2
            l2_uniques = sorted(df["category_level2"].unique())
            l2_encoder = {name: i for i, name in enumerate(l2_uniques)}

            # Level 3 (category_id is the unique identifier for L3)
            # We map category_id directly to 0..N-1
            l3_ids = sorted(df["category_id"].unique())
            l3_encoder = {cat_id: i for i, cat_id in enumerate(l3_ids)}

            # Apply encoding
            df["l1_idx"] = df["category_level1"].map(l1_encoder)
            df["l2_idx"] = df["category_level2"].map(l2_encoder)
            df["l3_idx"] = df["category_id"].map(l3_encoder)

            # Save cache
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            df.to_parquet(cache_path, index=False)

        # Populate internal dictionaries for fast lookup
        self.l3_inv_map = dict(zip(df["l3_idx"], df["category_id"]))

        # Create a fast lookup dictionary: category_id -> (l1, l2, l3)
        for _, row in df.iterrows():
            cat_id = int(row["category_id"])
            l1 = int(row["l1_idx"])
            l2 = int(row["l2_idx"])
            l3 = int(row["l3_idx"])
            self.cat_id_to_hierarchy[cat_id] = (l1, l2, l3)

    def get_labels(self, category_id):
        """
        Returns (l1_idx, l2_idx, l3_idx) for a given category_id.
        Returns (None, None, None) if category_id is unknown.
        """
        return self.cat_id_to_hierarchy.get(category_id, (None, None, None))

    def get_category_id(self, l3_idx):
        """
        Returns the original category_id for a given model prediction (l3_idx).
        """
        return self.l3_inv_map.get(l3_idx, None)


class BSONLoader:
    """
    Efficiently reads images from BSON files using byte offsets.
    """

    def __init__(self, bson_path):
        self.bson_path = bson_path

    def _get_val_size(self, type_byte, data, ptr):
        """Returns the size of a BSON value based on its type byte."""
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
        Parses a raw BSON document bytes to find the 'imgs' array and extract 'picture' binaries.
        """
        images = []
        ptr = 4  # Skip total size header
        length = len(data)

        while ptr < length - 1:
            type_byte = data[ptr]
            ptr += 1

            # Read Field Name
            name_end = data.find(b"\x00", ptr)
            # Optimization: Check if name is 'imgs' without full decode
            is_imgs = data[ptr:name_end] == b"imgs"
            ptr = name_end + 1

            if is_imgs and type_byte == 0x04:
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
                            # Optimization: Check if name is 'picture' without full decode
                            is_picture = data[dp:dname_end] == b"picture"
                            dp = dname_end + 1

                            if is_picture and dtype == 0x05:
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

    def read_images(self, offset, length):
        """
        Reads a product record from BSON and returns a list of decoded numpy images (BGR).

        Args:
            offset (int): Byte offset in the BSON file.
            length (int): Length of the BSON record.

        Returns:
            list[np.ndarray]: List of images (BGR format).
        """
        with open(self.bson_path, "rb") as f:
            f.seek(offset)
            data = f.read(length)

        img_binaries = self._extract_images_from_bytes(data)
        decoded_images = []

        for img_bytes in img_binaries:
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None:
                decoded_images.append(img)

        return decoded_images
