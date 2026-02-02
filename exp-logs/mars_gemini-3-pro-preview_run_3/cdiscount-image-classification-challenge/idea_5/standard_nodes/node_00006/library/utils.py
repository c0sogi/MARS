import os
import struct
import pickle
import pandas as pd
import numpy as np
import io
from PIL import Image
from library import config


class HierarchyEncoder:
    """
    Manages the mapping between product category IDs and their hierarchical levels (L1, L2, L3).
    Supports encoding targets for multi-task learning and decoding predictions for submission.
    """

    def __init__(self):
        self.l3_to_idx = {}
        self.idx_to_l3 = {}
        self.l3_to_l2_idx = {}
        self.l3_to_l1_idx = {}

        self.l1_classes = []
        self.l2_classes = []
        self.l3_classes = []

        self.num_l1 = 0
        self.num_l2 = 0
        self.num_l3 = 0

    def fit(self):
        """
        Reads the category names CSV and builds the hierarchical mappings.
        """
        df = pd.read_csv(config.CATEGORY_NAMES_PATH)

        # Sort unique values for deterministic mapping
        self.l1_classes = sorted(df["category_level1"].unique())
        self.l2_classes = sorted(df["category_level2"].unique())
        self.l3_classes = sorted(df["category_id"].unique())

        self.num_l1 = len(self.l1_classes)
        self.num_l2 = len(self.l2_classes)
        self.num_l3 = len(self.l3_classes)

        # Create mappings from Name/ID -> Integer Index
        l1_map = {name: i for i, name in enumerate(self.l1_classes)}
        l2_map = {name: i for i, name in enumerate(self.l2_classes)}
        l3_map = {cid: i for i, cid in enumerate(self.l3_classes)}

        self.l3_to_idx = l3_map
        self.idx_to_l3 = {i: cid for cid, i in l3_map.items()}

        # Build hierarchy vectors: Index is L3_idx, Value is corresponding L1/L2_idx
        self.l3_to_l1_idx = np.zeros(self.num_l3, dtype=np.int64)
        self.l3_to_l2_idx = np.zeros(self.num_l3, dtype=np.int64)

        # Iterate through the dataframe to fill relationships
        # Note: category_id is unique in the CSV
        for _, row in df.iterrows():
            cid = row["category_id"]
            if cid in l3_map:
                l3_idx = l3_map[cid]
                self.l3_to_l1_idx[l3_idx] = l1_map[row["category_level1"]]
                self.l3_to_l2_idx[l3_idx] = l2_map[row["category_level2"]]

    def save(self):
        """Saves the encoder state to disk."""
        with open(config.CATEGORY_ENCODER_PATH, "wb") as f:
            pickle.dump(self.__dict__, f)

    def load(self):
        """Loads the encoder state from disk."""
        with open(config.CATEGORY_ENCODER_PATH, "rb") as f:
            state = pickle.load(f)
            self.__dict__.update(state)

    def prepare(self, load_cached_data=True):
        """
        Ensures the encoder is ready. Tries to load from cache; otherwise fits and saves.
        """
        if load_cached_data and os.path.exists(config.CATEGORY_ENCODER_PATH):
            try:
                self.load()
                return
            except Exception:
                pass  # Fallback to fit if load fails

        self.fit()
        self.save()

    def transform(self, category_ids):
        """
        Converts a list/array of category_ids into indices for L3, L2, and L1.
        Returns: (l3_indices, l2_indices, l1_indices)
        """
        # Handle single item or list
        if np.isscalar(category_ids):
            category_ids = [category_ids]

        l3_indices = [
            self.l3_to_idx.get(cid, 0) for cid in category_ids
        ]  # Default to 0 if unknown (should not happen in train)
        l3_indices = np.array(l3_indices, dtype=np.int64)

        l2_indices = self.l3_to_l2_idx[l3_indices]
        l1_indices = self.l3_to_l1_idx[l3_indices]

        return l3_indices, l2_indices, l1_indices

    def inverse_transform(self, l3_indices):
        """
        Converts L3 integer indices back to original category_ids.
        """
        return [self.idx_to_l3.get(idx, self.l3_classes[0]) for idx in l3_indices]


class BSONReader:
    """
    Reads image data from BSON files using metadata offsets.
    """

    def __init__(self, bson_path):
        self.bson_path = bson_path

    def read_record(self, offset, length):
        """
        Reads a specific record from the BSON file and returns a list of image bytes.
        """
        with open(self.bson_path, "rb") as f:
            f.seek(offset)
            data = f.read(length)
        return self._extract_images_from_bson(data)

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
        Parses raw BSON bytes to extract 'picture' binary data from the 'imgs' array.
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


def load_metadata(split="train"):
    """
    Loads the metadata CSV for the specified split.
    """
    if split == "train":
        return pd.read_csv(config.TRAIN_META_PATH)
    elif split == "val":
        return pd.read_csv(config.VAL_META_PATH)
    elif split == "test":
        return pd.read_csv(config.TEST_META_PATH)
    else:
        raise ValueError(f"Unknown split: {split}. Must be 'train', 'val', or 'test'.")
