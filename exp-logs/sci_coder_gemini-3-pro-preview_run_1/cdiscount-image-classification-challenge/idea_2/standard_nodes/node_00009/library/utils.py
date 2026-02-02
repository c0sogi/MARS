import os
import struct
import io
import pandas as pd
import numpy as np
from PIL import Image
from library.config import Config

# ==========================================
# BSON Constants & Parsing Helpers
# ==========================================
BSON_TYPE_DOUBLE = 1
BSON_TYPE_STRING = 2
BSON_TYPE_OBJECT = 3
BSON_TYPE_ARRAY = 4
BSON_TYPE_BINARY = 5
BSON_TYPE_OBJECTID = 7
BSON_TYPE_BOOL = 8
BSON_TYPE_DATE = 9
BSON_TYPE_NULL = 10
BSON_TYPE_INT32 = 16
BSON_TYPE_INT64 = 18


def read_c_string(buffer, start):
    """Reads a C-style null-terminated string from buffer."""
    end = buffer.find(b"\x00", start)
    if end == -1:
        return None, -1
    return buffer[start:end].decode("utf-8", errors="ignore"), end + 1


def skip_bson_value(buffer, idx, type_byte):
    """Helper to skip a BSON value based on its type."""
    if idx >= len(buffer):
        return len(buffer)

    if type_byte == BSON_TYPE_DOUBLE:
        return idx + 8
    elif type_byte == BSON_TYPE_STRING:
        l = struct.unpack("<i", buffer[idx : idx + 4])[0]
        return idx + 4 + l
    elif type_byte == BSON_TYPE_OBJECT or type_byte == BSON_TYPE_ARRAY:
        l = struct.unpack("<i", buffer[idx : idx + 4])[0]
        return idx + l
    elif type_byte == BSON_TYPE_BINARY:
        l = struct.unpack("<i", buffer[idx : idx + 4])[0]
        return idx + 4 + 1 + l
    elif type_byte == BSON_TYPE_OBJECTID:
        return idx + 12
    elif type_byte == BSON_TYPE_BOOL:
        return idx + 1
    elif type_byte == BSON_TYPE_DATE:
        return idx + 8
    elif type_byte == BSON_TYPE_NULL:
        return idx
    elif type_byte == BSON_TYPE_INT32:
        return idx + 4
    elif type_byte == BSON_TYPE_INT64:
        return idx + 8
    else:
        # Unknown, jump to end
        return len(buffer)


def extract_images_from_bson(buffer):
    """
    Parses a BSON document buffer to extract image binary data from the 'imgs' array.
    Returns a list of PIL Images (RGB).
    """
    images = []
    idx = 0
    buf_len = len(buffer)

    # Heuristic: check if first 4 bytes match buffer length (standard BSON header)
    if buf_len >= 4:
        size = struct.unpack("<i", buffer[0:4])[0]
        if size == buf_len:
            idx = 4

    while idx < buf_len - 1:
        type_byte = buffer[idx]
        idx += 1

        name, idx = read_c_string(buffer, idx)
        if idx == -1:
            break

        # We are looking for 'imgs'
        if name == "imgs" and type_byte == BSON_TYPE_ARRAY:
            # Parse Array
            if idx + 4 > buf_len:
                break
            arr_len = struct.unpack("<i", buffer[idx : idx + 4])[0]
            arr_end = idx + arr_len

            # Enter array
            a_idx = idx + 4
            while a_idx < arr_end - 1:
                e_type = buffer[a_idx]
                a_idx += 1
                e_name, a_idx = read_c_string(buffer, a_idx)

                if e_type == BSON_TYPE_OBJECT:
                    # Inside the array element (dict with 'picture')
                    o_len = struct.unpack("<i", buffer[a_idx : a_idx + 4])[0]
                    o_end = a_idx + o_len

                    o_curr = a_idx + 4
                    while o_curr < o_end - 1:
                        p_type = buffer[o_curr]
                        o_curr += 1
                        p_name, o_curr = read_c_string(buffer, o_curr)

                        if p_name == "picture" and p_type == BSON_TYPE_BINARY:
                            b_len = struct.unpack("<i", buffer[o_curr : o_curr + 4])[0]
                            # subtype = buffer[o_curr + 4]
                            # Image data starts at o_curr + 5
                            img_data = buffer[o_curr + 5 : o_curr + 5 + b_len]

                            try:
                                img = Image.open(io.BytesIO(img_data)).convert("RGB")
                                images.append(img)
                            except Exception:
                                pass  # Skip corrupt images

                            o_curr += 5 + b_len
                        else:
                            o_curr = skip_bson_value(buffer, o_curr, p_type)

                    a_idx = o_end
                else:
                    a_idx = skip_bson_value(buffer, a_idx, e_type)

            return images
        else:
            idx = skip_bson_value(buffer, idx, type_byte)

    return images


def read_bson_images(bson_path, offset, length):
    """
    Reads a specific BSON record from disk and extracts images.

    Args:
        bson_path (str): Path to the .bson file.
        offset (int): Byte offset where the record starts.
        length (int): Length of the record in bytes.

    Returns:
        list[PIL.Image]: List of images found in the record.
    """
    with open(bson_path, "rb") as f:
        f.seek(offset)
        data = f.read(length)

    return extract_images_from_bson(data)


# ==========================================
# Category Hierarchy Manager
# ==========================================
class CategoryHierarchy:
    """
    Manages the mapping between raw category IDs and hierarchical integer indices
    (Level 1, Level 2, Level 3) for the model.
    """

    def __init__(self, load_cached_data=True):
        self.cache_path = os.path.join(Config.WORKING_DIR, "category_hierarchy.parquet")
        self.df_mapping = self._load_or_create_mapping(load_cached_data)

        # Create lookups
        # Map category_id (int) -> (l1_idx, l2_idx, l3_idx)
        self.id_to_indices = {}
        # Map l3_idx (int) -> category_id (int) [For submission]
        self.l3_idx_to_id = {}

        for _, row in self.df_mapping.iterrows():
            cat_id = int(row["category_id"])
            l1 = int(row["l1_idx"])
            l2 = int(row["l2_idx"])
            l3 = int(row["l3_idx"])

            self.id_to_indices[cat_id] = (l1, l2, l3)
            self.l3_idx_to_id[l3] = cat_id

    def _load_or_create_mapping(self, load_cached_data):
        # 1. Try to load cache
        if load_cached_data and os.path.exists(self.cache_path):
            try:
                return pd.read_parquet(self.cache_path)
            except Exception:
                pass  # Fallback to create

        # 2. Create from scratch
        df = pd.read_csv(Config.CATEGORY_NAMES)

        # Sort to ensure deterministic indexing
        df = df.sort_values("category_id").reset_index(drop=True)

        # Create mappings
        # Level 1
        l1_uniques = sorted(df["category_level1"].unique())
        l1_map = {name: idx for idx, name in enumerate(l1_uniques)}
        df["l1_idx"] = df["category_level1"].map(l1_map)

        # Level 2
        l2_uniques = sorted(df["category_level2"].unique())
        l2_map = {name: idx for idx, name in enumerate(l2_uniques)}
        df["l2_idx"] = df["category_level2"].map(l2_map)

        # Level 3 (Target) - Map category_id to 0..N-1
        # Since df is sorted by category_id, we can just use the index,
        # but let's be explicit to handle potential gaps if we filter later.
        # Here we map the sorted unique category_ids to 0..N
        cat_ids = sorted(df["category_id"].unique())
        l3_map = {cid: idx for idx, cid in enumerate(cat_ids)}
        df["l3_idx"] = df["category_id"].map(l3_map)

        # Save cache
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        df.to_parquet(self.cache_path, index=False)

        return df

    def get_hierarchy_indices(self, category_id):
        """
        Returns (l1_idx, l2_idx, l3_idx) for a given category_id.
        """
        return self.id_to_indices.get(category_id, (0, 0, 0))  # Default/Safety

    def get_l3_index(self, category_id):
        """
        Returns the target class index (0-5269) for a given category_id.
        """
        indices = self.id_to_indices.get(category_id)
        if indices:
            return indices[2]
        return 0

    def get_category_id_from_l3(self, l3_idx):
        """
        Returns the original category_id for a given model output index.
        """
        return self.l3_idx_to_id.get(l3_idx, 0)
