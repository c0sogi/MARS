import os
import struct
import io
import pandas as pd
import numpy as np
import torch
from library.config import Config

# BSON Constants
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
        # Unknown, jump to end to be safe
        return len(buffer)


def read_bson_images(file_handle, offset, length):
    """
    Reads images from an open BSON file handle at a specific offset.

    Args:
        file_handle: An open file object (rb mode).
        offset (int): Byte offset where the document starts.
        length (int): Length of the document in bytes.

    Returns:
        list: A list of binary image data (bytes).
    """
    file_handle.seek(offset)
    buffer = file_handle.read(length)

    images = []
    idx = 0
    buf_len = len(buffer)

    # Standard BSON starts with int32 size.
    # If the buffer passed is the full document, we skip the first 4 bytes.
    if buf_len >= 4:
        idx = 4

    while idx < buf_len - 1:
        type_byte = buffer[idx]
        idx += 1

        name, idx = read_c_string(buffer, idx)
        if idx == -1:
            break

        # We are looking for the 'imgs' array
        if name == "imgs" and type_byte == BSON_TYPE_ARRAY:
            if idx + 4 > buf_len:
                break
            # Array length
            arr_len = struct.unpack("<i", buffer[idx : idx + 4])[0]
            arr_end = idx + arr_len

            # Inside the array
            a_idx = idx + 4
            while a_idx < arr_end - 1:
                e_type = buffer[a_idx]
                a_idx += 1
                e_name, a_idx = read_c_string(buffer, a_idx)  # index "0", "1", etc.

                if e_type == BSON_TYPE_OBJECT:
                    o_len = struct.unpack("<i", buffer[a_idx : a_idx + 4])[0]
                    o_end = a_idx + o_len

                    o_curr = a_idx + 4
                    while o_curr < o_end - 1:
                        p_type = buffer[o_curr]
                        o_curr += 1
                        p_name, o_curr = read_c_string(buffer, o_curr)

                        if p_name == "picture" and p_type == BSON_TYPE_BINARY:
                            b_len = struct.unpack("<i", buffer[o_curr : o_curr + 4])[0]
                            # subtype is at o_curr + 4 (1 byte)
                            # Image data starts at o_curr + 5
                            img_data = buffer[o_curr + 5 : o_curr + 5 + b_len]
                            images.append(img_data)
                            o_curr += 5 + b_len
                        else:
                            o_curr = skip_bson_value(buffer, o_curr, p_type)
                    a_idx = o_end
                else:
                    a_idx = skip_bson_value(buffer, a_idx, e_type)

            # Found images, return immediately
            return images
        else:
            # Skip irrelevant fields
            idx = skip_bson_value(buffer, idx, type_byte)

    return images


class HierarchyManager:
    """
    Manages the mapping between fine-grained categories (Level 3) and
    their hierarchical parents (Level 2, Level 1).
    """

    def __init__(self, load_cached_data=True):
        self.mapping_df = self._load_or_create_mappings(load_cached_data)

        # Sort by category_id to ensure deterministic class_idx assignment (0 to N-1)
        self.mapping_df = self.mapping_df.sort_values("category_id").reset_index(
            drop=True
        )

        # Create mapping from real category_id to internal class index (0 to N-1)
        self.cat_id_to_class_idx = {
            cid: idx for idx, cid in enumerate(self.mapping_df["category_id"].values)
        }

        # Create lookup tensors for auxiliary targets
        # Index of tensor corresponds to class_idx (Level 3)
        # Value corresponds to parent class index (Level 1 or 2)
        self.class_idx_to_l1 = torch.tensor(
            self.mapping_df["l1_idx"].values, dtype=torch.long
        )
        self.class_idx_to_l2 = torch.tensor(
            self.mapping_df["l2_idx"].values, dtype=torch.long
        )

    def _load_or_create_mappings(self, load_cached_data):
        """
        Loads mappings from parquet or creates them from source files.
        """
        Config.setup()  # Ensure directories exist
        cache_path = Config.HIERARCHY_MAPPING_PATH

        if load_cached_data and os.path.exists(cache_path):
            try:
                df = pd.read_parquet(cache_path)
                # Simple validation
                expected_cols = [
                    "category_id",
                    "category_level1",
                    "category_level2",
                    "l1_idx",
                    "l2_idx",
                ]
                if all(col in df.columns for col in expected_cols):
                    return df
            except Exception:
                pass  # Fallback to create

        # Create from scratch
        return self._create_mappings(cache_path)

    def _create_mappings(self, cache_path):
        """
        Reads category_names.csv to build the hierarchy.
        """
        # Load Category Names
        df_cats = pd.read_csv(Config.CATEGORY_NAMES)

        # Ensure category_id is int
        df_cats = df_cats.dropna(subset=["category_id"])
        df_cats["category_id"] = df_cats["category_id"].astype(np.int64)

        # Encode Level 1
        l1_uniques = sorted(df_cats["category_level1"].astype(str).unique())
        l1_map = {val: i for i, val in enumerate(l1_uniques)}
        df_cats["l1_idx"] = df_cats["category_level1"].astype(str).map(l1_map)

        # Encode Level 2
        l2_uniques = sorted(df_cats["category_level2"].astype(str).unique())
        l2_map = {val: i for i, val in enumerate(l2_uniques)}
        df_cats["l2_idx"] = df_cats["category_level2"].astype(str).map(l2_map)

        # Select relevant columns
        result_df = df_cats[
            ["category_id", "category_level1", "category_level2", "l1_idx", "l2_idx"]
        ].copy()

        # Save to cache
        result_df.to_parquet(cache_path, index=False)

        return result_df

    def get_auxiliary_labels(self, class_indices):
        """
        Given a tensor of target class indices (Level 3), returns corresponding
        Level 1 and Level 2 indices.

        Args:
            class_indices (torch.Tensor): Tensor of shape (batch_size,) containing L3 indices (0..5269).

        Returns:
            tuple: (l1_indices, l2_indices)
        """
        # Move lookup tensors to same device as input
        device = class_indices.device
        l1_targets = self.class_idx_to_l1.to(device)[class_indices]
        l2_targets = self.class_idx_to_l2.to(device)[class_indices]
        return l1_targets, l2_targets

    def category_id_to_class_idx(self, category_id):
        """Converts a raw category_id to the internal model class index."""
        return self.cat_id_to_class_idx.get(category_id, -1)

    def class_idx_to_category_id(self, class_idx):
        """Converts an internal model class index back to raw category_id."""
        return self.mapping_df.iloc[class_idx]["category_id"]
