import os
import struct
import numpy as np
import pandas as pd
import cv2
import torch
from library.config import (
    TRAIN_BSON_PATH,
    CATEGORY_NAMES_PATH,
    WORKING_DIR,
    IMG_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    SEED,
)


def preprocess_image(img_bytes):
    """
    Decodes a byte string into a normalized PyTorch tensor.

    Args:
        img_bytes (bytes): Binary image data.

    Returns:
        torch.Tensor: Preprocessed image of shape (3, IMG_SIZE, IMG_SIZE).
                      Returns None if decoding fails.
    """
    if img_bytes is None or len(img_bytes) == 0:
        return None

    # Decode image from bytes
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

    if img is None:
        return None

    # Handle channels (convert Grayscale/Alpha to RGB)
    if len(img.shape) == 2:  # Grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:  # RGBA
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:  # BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

    # Normalize
    img = img.astype(np.float32) / 255.0

    # Standardize using ImageNet stats
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    img = (img - mean) / std

    # Transpose to (C, H, W)
    img = img.transpose(2, 0, 1)

    return torch.from_numpy(img)


class HierarchyEncoder:
    """
    Manages the mapping between raw category_ids and hierarchical integer labels.
    """

    def __init__(self, load_cached_data=True):
        self.cache_path = os.path.join(WORKING_DIR, "hierarchy_map.parquet")
        self.mapping_df = self._load_or_build_mapping(load_cached_data)

        # Create fast lookup dictionary
        # Key: category_id (int), Value: (l1_idx, l2_idx, l3_idx)
        self.lookup = {}
        for _, row in self.mapping_df.iterrows():
            self.lookup[row["category_id"]] = (
                row["l1_idx"],
                row["l2_idx"],
                row["l3_idx"],
            )

        # Store class counts
        self.num_l1 = self.mapping_df["l1_idx"].max() + 1
        self.num_l2 = self.mapping_df["l2_idx"].max() + 1
        self.num_l3 = self.mapping_df["l3_idx"].max() + 1

        # Reverse mapping for L3 (needed for submission)
        # Key: l3_idx, Value: category_id
        self.l3_to_cat_id = {v[2]: k for k, v in self.lookup.items()}

    def _load_or_build_mapping(self, load_cached_data):
        if load_cached_data and os.path.exists(self.cache_path):
            try:
                df = pd.read_parquet(self.cache_path)
                return df
            except Exception:
                pass  # Fallback to build

        # Build from source
        if not os.path.exists(CATEGORY_NAMES_PATH):
            raise FileNotFoundError(
                f"Category names file not found at {CATEGORY_NAMES_PATH}"
            )

        df = pd.read_csv(CATEGORY_NAMES_PATH)

        # Ensure category_id is int
        df["category_id"] = df["category_id"].astype(int)

        # Factorize levels to get 0..N-1 indices
        # We sort to ensure deterministic mapping

        # Level 1
        l1_cats = sorted(df["category_level1"].unique())
        l1_map = {name: i for i, name in enumerate(l1_cats)}
        df["l1_idx"] = df["category_level1"].map(l1_map)

        # Level 2
        l2_cats = sorted(df["category_level2"].unique())
        l2_map = {name: i for i, name in enumerate(l2_cats)}
        df["l2_idx"] = df["category_level2"].map(l2_map)

        # Level 3 (Target) - Map raw category_id to contiguous index
        # Note: category_level3 names are not unique across hierarchy,
        # but category_id is unique. We map category_id directly.
        l3_ids = sorted(df["category_id"].unique())
        l3_map = {cid: i for i, cid in enumerate(l3_ids)}
        df["l3_idx"] = df["category_id"].map(l3_map)

        # Save cache
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        df.to_parquet(self.cache_path, index=False)

        return df

    def get_labels(self, category_id):
        """
        Returns (l1_idx, l2_idx, l3_idx) for a given raw category_id.
        """
        return self.lookup.get(category_id, (-1, -1, -1))

    def get_category_id(self, l3_idx):
        """
        Returns raw category_id for a given model prediction index (l3_idx).
        """
        return self.l3_to_cat_id.get(l3_idx, -1)


class BSONIterator:
    """
    Handles random access to BSON file to extract images for specific records.
    """

    def __init__(self, bson_path, metadata_df):
        self.bson_path = bson_path
        self.metadata = metadata_df
        self.file_handle = None

    def _open_file(self):
        if self.file_handle is None:
            self.file_handle = open(self.bson_path, "rb")

    def __del__(self):
        if self.file_handle is not None:
            self.file_handle.close()

    def get_images(self, idx):
        """
        Retrieves raw image bytes list for the record at metadata index `idx`.
        """
        self._open_file()

        row = self.metadata.iloc[idx]
        offset = row["bson_offset"]
        length = row["bson_length"]

        self.file_handle.seek(offset)
        doc_data = self.file_handle.read(length)

        return self._extract_images_from_bson(doc_data)

    def _get_val_size(self, type_byte, data, ptr):
        """Helper to determine BSON value size."""
        if type_byte == 0x01:
            return 8  # double
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
        elif type_byte == 0x07:
            return 12  # objectid
        elif type_byte == 0x08:
            return 1  # boolean
        elif type_byte == 0x09:
            return 8  # utc datetime
        elif type_byte == 0x0A:
            return 0  # null
        elif type_byte == 0x10:
            return 4  # int32
        elif type_byte == 0x12:
            return 8  # int64
        else:
            return 0

    def _extract_images_from_bson(self, data):
        """
        Parses BSON bytes to extract 'picture' binary fields from 'imgs' array.
        """
        images = []
        ptr = 4  # Skip total size
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
                # Found imgs array
                arr_len = struct.unpack("<i", data[ptr : ptr + 4])[0]
                arr_end = ptr + arr_len

                # Enter Array
                ap = ptr + 4
                while ap < arr_end - 1:
                    etype = data[ap]
                    ap += 1

                    # Skip index key ("0", "1", etc)
                    ename_end = data.find(b"\x00", ap)
                    ap = ename_end + 1

                    if etype == 0x03:  # Document
                        doc_len = struct.unpack("<i", data[ap : ap + 4])[0]
                        doc_end = ap + doc_len

                        # Enter Image Document
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
                                v_len = self._get_val_size(dtype, data, dp)
                                dp += v_len
                        ap += doc_len
                    else:
                        v_len = self._get_val_size(etype, data, ap)
                        ap += v_len

                # Found what we needed, can return early if we assume only one imgs array
                return images

            else:
                v_len = self._get_val_size(type_byte, data, ptr)
                ptr += v_len

        return images
