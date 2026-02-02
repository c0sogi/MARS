import os
import struct
import pandas as pd
import numpy as np
import cv2
import torch
import random
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class BSONImageLoader:
    """
    Efficiently reads and decodes images from a BSON file using byte offsets.
    Maintains an open file handle to avoid overhead from repeated opens.
    """

    def __init__(self, bson_path):
        self.bson_path = bson_path
        self.file_handle = open(bson_path, "rb")

    def __del__(self):
        if hasattr(self, "file_handle") and self.file_handle:
            self.file_handle.close()

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
        Parses raw BSON bytes to find 'imgs' array and extract 'picture' binaries.
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

    def load_images(self, offset, length):
        """
        Reads BSON record at offset/length and returns a list of decoded numpy images (BGR).
        """
        self.file_handle.seek(offset)
        doc_data = self.file_handle.read(length)

        img_binaries = self._extract_images_from_bytes(doc_data)
        decoded_images = []

        for img_bytes in img_binaries:
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
            if img is not None:
                decoded_images.append(img)

        return decoded_images


class HierarchyMapper:
    """
    Manages the mapping between raw category IDs and hierarchical indices (L1, L2, L3).
    Implements caching to Parquet to ensure consistency and speed.
    """

    def __init__(self, category_names_path):
        self.category_names_path = category_names_path
        self.l3_to_l1_map = None
        self.l3_to_l2_map = None
        self.raw_to_l3_map = None
        self.l3_to_raw_map = None

    def process(self, load_cached=True):
        """
        Loads or creates the hierarchy mappings.

        Returns:
            df_mapping (pd.DataFrame): DataFrame containing the mappings.
        """
        cache_path = Config.HIERARCHY_MAPPING

        if load_cached and os.path.exists(cache_path):
            df_mapping = pd.read_parquet(cache_path)
        else:
            # Create from scratch
            df_cats = pd.read_csv(self.category_names_path)

            # Ensure deterministic ordering by sorting
            # Level 1
            l1_names = sorted(df_cats["category_level1"].unique())
            l1_map = {name: i for i, name in enumerate(l1_names)}

            # Level 2
            l2_names = sorted(df_cats["category_level2"].unique())
            l2_map = {name: i for i, name in enumerate(l2_names)}

            # Level 3 (Target) - Sort by category_id for stable indexing
            l3_ids = sorted(df_cats["category_id"].unique())
            l3_map = {cid: i for i, cid in enumerate(l3_ids)}

            # Apply mappings
            df_cats["l1_idx"] = df_cats["category_level1"].map(l1_map)
            df_cats["l2_idx"] = df_cats["category_level2"].map(l2_map)
            df_cats["l3_idx"] = df_cats["category_id"].map(l3_map)

            # Select relevant columns
            df_mapping = df_cats[["category_id", "l1_idx", "l2_idx", "l3_idx"]].copy()

            # Save to cache
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            df_mapping.to_parquet(cache_path, index=False)

        # Build internal lookup structures
        # We need fast lookups:
        # 1. raw_id -> l3_idx (for training labels)
        # 2. l3_idx -> l1_idx, l2_idx (for hierarchical loss)
        # 3. l3_idx -> raw_id (for submission)

        self.raw_to_l3_map = dict(zip(df_mapping["category_id"], df_mapping["l3_idx"]))
        self.l3_to_raw_map = dict(zip(df_mapping["l3_idx"], df_mapping["category_id"]))

        # Create arrays where index is l3_idx and value is parent idx
        max_l3 = df_mapping["l3_idx"].max()
        self.l3_to_l1_map = np.zeros(max_l3 + 1, dtype=np.int64)
        self.l3_to_l2_map = np.zeros(max_l3 + 1, dtype=np.int64)

        # Fill arrays
        # Ensure we iterate in order of l3_idx to fill correctly
        df_sorted = df_mapping.sort_values("l3_idx")
        self.l3_to_l1_map[df_sorted["l3_idx"].values] = df_sorted["l1_idx"].values
        self.l3_to_l2_map[df_sorted["l3_idx"].values] = df_sorted["l2_idx"].values

        return df_mapping

    def get_hierarchical_labels(self, raw_category_id):
        """
        Returns (l1_idx, l2_idx, l3_idx) for a given raw category ID.
        """
        if self.raw_to_l3_map is None:
            raise RuntimeError(
                "HierarchyMapper.process() must be called before querying."
            )

        l3_idx = self.raw_to_l3_map.get(raw_category_id)
        if l3_idx is None:
            return None, None, None

        l1_idx = self.l3_to_l1_map[l3_idx]
        l2_idx = self.l3_to_l2_map[l3_idx]

        return l1_idx, l2_idx, l3_idx

    def get_raw_category_id(self, l3_idx):
        """
        Returns the raw category ID for a given model prediction index (L3).
        """
        if self.l3_to_raw_map is None:
            raise RuntimeError(
                "HierarchyMapper.process() must be called before querying."
            )
        return self.l3_to_raw_map.get(l3_idx)
