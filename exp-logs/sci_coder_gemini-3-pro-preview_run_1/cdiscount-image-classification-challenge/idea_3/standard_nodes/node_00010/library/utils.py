import os
import struct
import io
import random
import numpy as np
import pandas as pd
import torch
from PIL import Image
import torchvision.transforms.functional as TF
from library.config import Config

# Set random seeds for reproducibility
random.seed(Config.SEED)
np.random.seed(Config.SEED)
torch.manual_seed(Config.SEED)

# BSON Type Constants
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


def extract_images_from_bson(buffer, transform=None):
    """
    Parses a BSON document buffer to extract image binary data from the 'imgs' array
    and converts them into image tensors.

    Args:
        buffer (bytes): The binary content of the BSON document.
        transform (callable, optional): A function/transform that takes in an PIL image
                                        and returns a transformed version.

    Returns:
        list[torch.Tensor]: A list of image tensors.
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

        name, idx = _read_c_string(buffer, idx)
        if idx == -1:
            break

        # We are looking for 'imgs'
        if name == "imgs" and type_byte == BSON_TYPE_ARRAY:
            # Parse Array (which is a BSON object)
            if idx + 4 > buf_len:
                break
            arr_len = struct.unpack("<i", buffer[idx : idx + 4])[0]
            arr_end = idx + arr_len

            # Enter array
            a_idx = idx + 4
            while a_idx < arr_end - 1:
                e_type = buffer[a_idx]
                a_idx += 1
                e_name, a_idx = _read_c_string(buffer, a_idx)  # index "0", "1", etc.

                if e_type == BSON_TYPE_OBJECT:
                    # Inside the array element (dict with 'picture')
                    o_len = struct.unpack("<i", buffer[a_idx : a_idx + 4])[0]
                    o_end = a_idx + o_len

                    o_curr = a_idx + 4
                    while o_curr < o_end - 1:
                        p_type = buffer[o_curr]
                        o_curr += 1
                        p_name, o_curr = _read_c_string(buffer, o_curr)

                        if p_name == "picture" and p_type == BSON_TYPE_BINARY:
                            b_len = struct.unpack("<i", buffer[o_curr : o_curr + 4])[0]
                            # subtype = buffer[o_curr + 4] # usually 0
                            # Image data starts at o_curr + 5
                            img_data = buffer[o_curr + 5 : o_curr + 5 + b_len]

                            # Convert to Tensor
                            try:
                                img = Image.open(io.BytesIO(img_data)).convert("RGB")
                                if transform:
                                    img_tensor = transform(img)
                                else:
                                    img_tensor = TF.to_tensor(img)
                                images.append(img_tensor)
                            except Exception:
                                # Skip corrupt images
                                pass

                            o_curr += 5 + b_len
                        else:
                            # Skip value
                            o_curr = _skip_bson_value(buffer, o_curr, p_type)

                    a_idx = o_end
                else:
                    a_idx = _skip_bson_value(buffer, a_idx, e_type)

            # We found 'imgs', we can stop parsing the main doc
            return images
        else:
            # Skip value
            idx = _skip_bson_value(buffer, idx, type_byte)

    return images


def load_category_hierarchy(load_cached_data=True):
    """
    Loads the category hierarchy mapping.
    Maps Level 3 category IDs (original) to contiguous indices for Level 1, 2, and 3.

    Args:
        load_cached_data (bool): If True, attempts to load from parquet cache.

    Returns:
        pd.DataFrame: DataFrame indexed by original 'category_id', containing:
                      ['l1_idx', 'l2_idx', 'l3_idx', 'category_level1', 'category_level2']
    """
    cache_path = Config.HIERARCHY_CACHE_PATH

    # 1. Try Loading Cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            # Ensure the index is the category_id
            if df.index.name != "category_id":
                df = df.set_index("category_id")
            return df
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from Scratch
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    # Load raw category names
    df_cats = pd.read_csv(Config.CATEGORY_NAMES)

    # Sort by category_id to ensure deterministic mapping for L3
    df_cats = df_cats.sort_values("category_id").reset_index(drop=True)

    # Create Mappings
    # Level 3 (Fine-grained): Map original ID to 0..N-1 based on sorted order
    # Since we sorted, the index is the label.
    df_cats["l3_idx"] = df_cats.index

    # Level 1 (Coarse): Factorize
    # We sort by name to ensure deterministic integer assignment
    l1_names = sorted(df_cats["category_level1"].unique())
    l1_map = {name: i for i, name in enumerate(l1_names)}
    df_cats["l1_idx"] = df_cats["category_level1"].map(l1_map)

    # Level 2 (Mid): Factorize
    l2_names = sorted(df_cats["category_level2"].unique())
    l2_map = {name: i for i, name in enumerate(l2_names)}
    df_cats["l2_idx"] = df_cats["category_level2"].map(l2_map)

    # Set index for fast lookup
    df_cats = df_cats.set_index("category_id")

    # Validation
    if len(df_cats) != Config.NUM_CLASSES_L3:
        print(
            f"Warning: Number of L3 classes ({len(df_cats)}) matches Config ({Config.NUM_CLASSES_L3})"
        )

    # Save to cache
    df_cats.to_parquet(cache_path)

    return df_cats


def calculate_accuracy(output, target):
    """
    Computes the accuracy of predictions.

    Args:
        output (torch.Tensor): Logits or probabilities of shape (B, C).
        target (torch.Tensor): Ground truth indices of shape (B).

    Returns:
        float: Accuracy score (0.0 to 1.0).
    """
    with torch.no_grad():
        # Get the index of the max log-probability
        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        return correct / target.size(0)
