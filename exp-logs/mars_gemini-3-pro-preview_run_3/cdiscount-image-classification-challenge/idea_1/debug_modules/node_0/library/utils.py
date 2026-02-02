import os
import struct
import pandas as pd
import numpy as np
import torch
from library.config import (
    CACHE_DIR,
    CATEGORY_NAMES,
    TRAIN_META_PATH,
    NUM_CLASSES,
    DEVICE,
)

# ==========================================
# BSON PARSING UTILITIES
# ==========================================


def _get_val_size(type_byte, data, ptr):
    """
    Helper function to determine the size of a BSON value based on its type.
    """
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
    Parses a raw BSON document byte string to find the 'imgs' array
    and extract 'picture' binaries.

    Args:
        data (bytes): The raw bytes of a BSON document.

    Returns:
        list[bytes]: A list of binary strings, each representing an image (JPEG).
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
            if ptr + 4 > length:
                break
            arr_len = struct.unpack("<i", data[ptr : ptr + 4])[0]
            arr_end = ptr + arr_len

            # Enter Array (skip length int)
            ap = ptr + 4
            while ap < arr_end - 1:
                if ap >= length:
                    break
                etype = data[ap]
                ap += 1

                # Array keys are "0", "1"... skip them
                ename_end = data.find(b"\x00", ap)
                if ename_end == -1:
                    break
                ap = ename_end + 1

                if etype == 0x03:  # Document (Image container)
                    if ap + 4 > length:
                        break
                    doc_len = struct.unpack("<i", data[ap : ap + 4])[0]
                    doc_end = ap + doc_len

                    # Enter Document
                    dp = ap + 4
                    while dp < doc_end - 1:
                        if dp >= length:
                            break
                        dtype = data[dp]
                        dp += 1

                        dname_end = data.find(b"\x00", dp)
                        if dname_end == -1:
                            break
                        dname = data[dp:dname_end].decode("utf-8", errors="ignore")
                        dp = dname_end + 1

                        if dname == "picture" and dtype == 0x05:
                            # Found picture binary
                            if dp + 4 > length:
                                break
                            bin_len = struct.unpack("<i", data[dp : dp + 4])[0]
                            # subtype is at dp+4, data starts at dp+5
                            if dp + 5 + bin_len > length:
                                break
                            img_bytes = data[dp + 5 : dp + 5 + bin_len]
                            images.append(img_bytes)
                            dp += 4 + 1 + bin_len
                        else:
                            # Skip other fields in image doc
                            v_len = _get_val_size(dtype, data, dp)
                            dp += v_len

                    ap += doc_len
                else:
                    v_len = _get_val_size(etype, data, ap)
                    ap += v_len

            ptr += arr_len
        else:
            # Skip this field
            v_len = _get_val_size(type_byte, data, ptr)
            ptr += v_len

    return images


# ==========================================
# DATA PROCESSING & CACHING
# ==========================================


def get_category_mapping(load_cached_data=True):
    """
    Creates a consistent mapping between category_id and a contiguous index (0 to NUM_CLASSES-1).
    Uses caching to ensure consistency and speed.

    Args:
        load_cached_data (bool): If True, attempts to load from cache.

    Returns:
        tuple: (id_to_idx (dict), idx_to_id (np.ndarray))
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, "category_ids.npy")

    if load_cached_data and os.path.exists(cache_path):
        category_ids = np.load(cache_path)
    else:
        # Load all categories from the definition file
        df_cats = pd.read_csv(CATEGORY_NAMES)
        category_ids = df_cats["category_id"].unique()
        category_ids.sort()  # Ensure deterministic order

        # Save to cache
        np.save(cache_path, category_ids)

    # Create mapping dictionary
    id_to_idx = {cat_id: idx for idx, cat_id in enumerate(category_ids)}

    return id_to_idx, category_ids


def calculate_class_weights(metadata_path=TRAIN_META_PATH, load_cached_data=True):
    """
    Calculates inverse frequency weights for class imbalance handling.

    Args:
        metadata_path (str): Path to the training metadata CSV.
        load_cached_data (bool): If True, attempts to load from cache.

    Returns:
        torch.Tensor: A tensor of shape (NUM_CLASSES,) containing weights.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, "class_weights.npy")

    if load_cached_data and os.path.exists(cache_path):
        weights = np.load(cache_path)
    else:
        # Load metadata
        df = pd.read_csv(metadata_path)

        # Get mapping
        id_to_idx, _ = get_category_mapping(load_cached_data=True)

        # Count frequencies
        counts = df["category_id"].value_counts()

        # Initialize weights array
        num_classes = len(id_to_idx)
        class_counts = np.zeros(num_classes)

        for cat_id, count in counts.items():
            if cat_id in id_to_idx:
                class_counts[id_to_idx[cat_id]] = count

        # Handle classes with 0 samples (should not happen in valid split, but for safety)
        # We replace 0 with 1 to avoid division by zero
        class_counts = np.where(class_counts == 0, 1, class_counts)

        # Calculate weights: n_samples / (n_classes * n_samples_j)
        # This is the 'balanced' heuristic from sklearn
        n_samples = len(df)
        weights = n_samples / (num_classes * class_counts)

        # Save to cache
        np.save(cache_path, weights)

    return torch.tensor(weights, dtype=torch.float32)


def get_accuracy(y_pred, y_true):
    """
    Calculates classification accuracy.

    Args:
        y_pred (torch.Tensor or np.ndarray): Predictions (indices).
        y_true (torch.Tensor or np.ndarray): Ground truth (indices).

    Returns:
        float: Accuracy score.
    """
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()

    correct = (y_pred == y_true).sum()
    total = len(y_true)

    if total == 0:
        return 0.0

    return float(correct) / total
