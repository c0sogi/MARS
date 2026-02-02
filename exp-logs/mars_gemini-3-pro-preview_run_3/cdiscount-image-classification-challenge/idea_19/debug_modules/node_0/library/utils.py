import os
import struct
import pandas as pd
import numpy as np
import cv2
import torch
from library.config import Config


class HierarchyMapper:
    """
    Manages the mapping between raw category IDs and the hierarchical
    indices (Level 1, Level 2, Level 3) used by the model.
    """

    def __init__(self, load_cached_data=True):
        self.l3_id_to_idx = {}
        self.l3_idx_to_id = {}
        self.l3_to_l2_map = None  # Array where index is L3_idx, value is L2_idx
        self.l3_to_l1_map = None  # Array where index is L3_idx, value is L1_idx

        self.num_classes_l1 = 0
        self.num_classes_l2 = 0
        self.num_classes_l3 = 0

        self._initialize(load_cached_data)

    def _initialize(self, load_cached_data):
        cache_path = Config.HIERARCHY_MAPPING

        if load_cached_data and os.path.exists(cache_path):
            try:
                df = pd.read_parquet(cache_path)
                self._build_maps_from_df(df)
                return
            except Exception as e:
                print(f"Failed to load cached hierarchy mapping: {e}. Recomputing...")

        # Compute from scratch
        df_cats = pd.read_csv(Config.CATEGORY_NAMES)

        # Sort to ensure deterministic indexing
        df_cats = df_cats.sort_values("category_id").reset_index(drop=True)

        # Create encodings
        # Level 1
        l1_uniques = sorted(df_cats["category_level1"].unique())
        l1_map = {name: i for i, name in enumerate(l1_uniques)}
        df_cats["l1_idx"] = df_cats["category_level1"].map(l1_map)

        # Level 2
        l2_uniques = sorted(df_cats["category_level2"].unique())
        l2_map = {name: i for i, name in enumerate(l2_uniques)}
        df_cats["l2_idx"] = df_cats["category_level2"].map(l2_map)

        # Level 3 (Target) - Map raw ID to 0..N-1
        # We assume the dataframe index after sorting is the L3 index
        df_cats["l3_idx"] = df_cats.index

        # Save cache
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        df_cats.to_parquet(cache_path)

        self._build_maps_from_df(df_cats)

    def _build_maps_from_df(self, df):
        self.num_classes_l1 = df["l1_idx"].max() + 1
        self.num_classes_l2 = df["l2_idx"].max() + 1
        self.num_classes_l3 = len(df)

        # Dictionaries for conversion
        self.l3_id_to_idx = dict(zip(df["category_id"], df["l3_idx"]))
        self.l3_idx_to_id = dict(zip(df["l3_idx"], df["category_id"]))

        # Arrays for fast lookup of parent categories
        # Index of array is L3_idx, value is parent L_idx
        self.l3_to_l1_map = np.zeros(self.num_classes_l3, dtype=np.int64)
        self.l3_to_l2_map = np.zeros(self.num_classes_l3, dtype=np.int64)

        self.l3_to_l1_map[df["l3_idx"].values] = df["l1_idx"].values
        self.l3_to_l2_map[df["l3_idx"].values] = df["l2_idx"].values

        # Convert to torch tensors for use in Datasets/Loss
        self.l3_to_l1_tensor = torch.from_numpy(self.l3_to_l1_map)
        self.l3_to_l2_tensor = torch.from_numpy(self.l3_to_l2_map)

    def get_parent_labels(self, l3_indices):
        """
        Given a tensor of Level 3 indices, returns corresponding Level 1 and Level 2 indices.
        """
        if isinstance(l3_indices, torch.Tensor):
            device = l3_indices.device
            l1 = self.l3_to_l1_tensor.to(device)[l3_indices]
            l2 = self.l3_to_l2_tensor.to(device)[l3_indices]
            return l1, l2
        else:
            # Numpy fallback
            l1 = self.l3_to_l1_map[l3_indices]
            l2 = self.l3_to_l2_map[l3_indices]
            return l1, l2


def get_val_size(type_byte, data, ptr):
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


def extract_images_from_bson(data):
    """
    Parses a raw BSON document byte string to extract 'picture' binaries.
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
                            v_len = get_val_size(dtype, data, dp)
                            dp += v_len

                    ap += doc_len
                else:
                    v_len = get_val_size(etype, data, ap)
                    ap += v_len

            ptr += arr_len
        else:
            # Skip this field
            v_len = get_val_size(type_byte, data, ptr)
            ptr += v_len

    return images


def process_image(img_bytes, target_size=Config.IMG_SIZE):
    """
    Decodes a byte string into a numpy array and resizes it.
    Returns: np.ndarray (H, W, 3) or None if decoding fails.
    """
    if not img_bytes:
        return None

    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if target_size:
        img = cv2.resize(
            img, (target_size, target_size), interpolation=cv2.INTER_LINEAR
        )

    return img


class BSONImageLoader:
    """
    Helper class to read BSON records from disk.
    Can be used as a context manager or with an external file handle.
    """

    def __init__(self, bson_path):
        self.bson_path = bson_path
        self.file_handle = None

    def open(self):
        if self.file_handle is None:
            self.file_handle = open(self.bson_path, "rb")

    def close(self):
        if self.file_handle is not None:
            self.file_handle.close()
            self.file_handle = None

    def read_images(self, offset, length):
        """
        Reads a record at the given offset and extracts images.
        """
        close_after = False
        if self.file_handle is None:
            self.open()
            close_after = True

        try:
            self.file_handle.seek(offset)
            data = self.file_handle.read(length)
            raw_images = extract_images_from_bson(data)

            processed_images = []
            for raw_img in raw_images:
                img = process_image(raw_img)
                if img is not None:
                    processed_images.append(img)

            return processed_images

        finally:
            if close_after:
                self.close()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
