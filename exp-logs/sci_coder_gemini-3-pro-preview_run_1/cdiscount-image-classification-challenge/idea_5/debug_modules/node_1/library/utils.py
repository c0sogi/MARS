import os
import struct
import io
import pandas as pd
import numpy as np
from PIL import Image
from library.config import get_hierarchy_mappings

# ==========================================
# BSON CONSTANTS & PARSING HELPERS
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


def _read_c_string(buffer, start):
    """Reads a C-style null-terminated string from buffer."""
    end = buffer.find(b"\x00", start)
    if end == -1:
        return None, -1
    return buffer[start:end].decode("utf-8", errors="ignore"), end + 1


def _skip_bson_value(buffer, idx, type_byte):
    """Helper to skip a BSON value based on its type."""
    buf_len = len(buffer)
    if idx >= buf_len:
        return buf_len

    if type_byte == BSON_TYPE_DOUBLE:
        return idx + 8
    elif type_byte == BSON_TYPE_STRING:
        if idx + 4 > buf_len:
            return buf_len
        l = struct.unpack("<i", buffer[idx : idx + 4])[0]
        return idx + 4 + l
    elif type_byte == BSON_TYPE_OBJECT or type_byte == BSON_TYPE_ARRAY:
        if idx + 4 > buf_len:
            return buf_len
        l = struct.unpack("<i", buffer[idx : idx + 4])[0]
        return idx + l
    elif type_byte == BSON_TYPE_BINARY:
        if idx + 4 > buf_len:
            return buf_len
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
        return buf_len


def extract_images_from_bson_data(doc_bytes):
    """
    Parses a BSON document bytes object to extract image binary data from the 'imgs' array.
    Returns a list of PIL Images.
    """
    images = []
    idx = 0
    buf_len = len(doc_bytes)

    # Skip size (4 bytes) if present at start and matches length
    # Standard BSON starts with int32 size
    if buf_len >= 4:
        size = struct.unpack("<i", doc_bytes[0:4])[0]
        if size == buf_len:
            idx = 4

    while idx < buf_len - 1:
        type_byte = doc_bytes[idx]
        idx += 1

        name, idx = _read_c_string(doc_bytes, idx)
        if idx == -1:
            break

        if name == "imgs" and type_byte == BSON_TYPE_ARRAY:
            # Parse Array
            if idx + 4 > buf_len:
                break
            arr_len = struct.unpack("<i", doc_bytes[idx : idx + 4])[0]
            arr_end = idx + arr_len

            a_idx = idx + 4
            while a_idx < arr_end - 1:
                e_type = doc_bytes[a_idx]
                a_idx += 1
                e_name, a_idx = _read_c_string(doc_bytes, a_idx)

                if e_type == BSON_TYPE_OBJECT:
                    o_len = struct.unpack("<i", doc_bytes[a_idx : a_idx + 4])[0]
                    o_end = a_idx + o_len
                    o_curr = a_idx + 4

                    while o_curr < o_end - 1:
                        p_type = doc_bytes[o_curr]
                        o_curr += 1
                        p_name, o_curr = _read_c_string(doc_bytes, o_curr)

                        if p_name == "picture" and p_type == BSON_TYPE_BINARY:
                            b_len = struct.unpack("<i", doc_bytes[o_curr : o_curr + 4])[
                                0
                            ]
                            # subtype = doc_bytes[o_curr + 4] # Skip subtype byte
                            img_data = doc_bytes[o_curr + 5 : o_curr + 5 + b_len]

                            try:
                                img = Image.open(io.BytesIO(img_data))
                                img.load()  # Force load image data into memory
                                images.append(img)
                            except Exception:
                                pass

                            o_curr += 5 + b_len
                        else:
                            o_curr = _skip_bson_value(doc_bytes, o_curr, p_type)
                    a_idx = o_end
                else:
                    a_idx = _skip_bson_value(doc_bytes, a_idx, e_type)
            return images
        else:
            idx = _skip_bson_value(doc_bytes, idx, type_byte)

    return images


def read_bson_images(file_obj, offset, length):
    """
    Reads a BSON record from an open file object at the given offset and length,
    then extracts images.

    Args:
        file_obj: Open file object (rb mode).
        offset (int): Byte offset in the file.
        length (int): Byte length of the record.

    Returns:
        list[PIL.Image]: List of images found in the product record.
    """
    file_obj.seek(offset)
    doc_bytes = file_obj.read(length)
    return extract_images_from_bson_data(doc_bytes)


# ==========================================
# HIERARCHY MAPPER
# ==========================================
class HierarchyMapper:
    """
    Handles mapping between category_ids and hierarchical levels (L1, L2, L3).
    Wraps the configuration utility to provide fast lookups.
    """

    def __init__(self, load_cached_data=True):
        self.mappings_df, self.stats = get_hierarchy_mappings(
            load_cached_data=load_cached_data
        )

        # Create fast lookup dictionary: category_id -> {l1, l2, l3}
        # We convert indices to standard python ints for compatibility
        self.lookup = {}
        for _, row in self.mappings_df.iterrows():
            self.lookup[row["category_id"]] = {
                "l1": int(row["l1_idx"]),
                "l2": int(row["l2_idx"]),
                "l3": int(row["l3_idx"]),
            }

    def get_labels(self, category_id):
        """
        Returns the hierarchical labels for a given category_id.

        Args:
            category_id (int): The product category ID.

        Returns:
            dict: {'l1': int, 'l2': int, 'l3': int} or None if not found.
        """
        return self.lookup.get(category_id)

    def get_num_classes(self):
        """
        Returns the number of classes for each level.

        Returns:
            dict: {'num_classes_l1': int, 'num_classes_l2': int, 'num_classes_l3': int}
        """
        return self.stats
