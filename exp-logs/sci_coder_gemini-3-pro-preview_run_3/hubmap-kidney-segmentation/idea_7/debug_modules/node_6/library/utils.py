import os
import json
import random
import numpy as np
import cv2
import torch
import pandas as pd
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def rle_encode(img):
    """
    Encodes a binary mask into Run-Length Encoding (RLE).
    Pixels are numbered from top to bottom, then left to right (Fortran/Column-major order).

    Args:
        img (np.ndarray): Binary mask (0 or 1).

    Returns:
        str: RLE string.
    """
    # Flatten in column-major order
    pixels = img.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle, shape):
    """
    Decodes an RLE string into a binary mask.

    Args:
        mask_rle (str): RLE string.
        shape (tuple): (height, width) of the mask.

    Returns:
        np.ndarray: Binary mask.
    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order="F")


def get_tissue_mask(
    image_id, width, height, anatomical_json_path, load_cached_data=True
):
    """
    Generates a binary mask for the tissue region (Cortex + Medulla) from anatomical polygons.
    Implements caching and Fail-Open logic.

    Args:
        image_id (str): Unique identifier for the image.
        width (int): Width of the image.
        height (int): Height of the image.
        anatomical_json_path (str): Path to the anatomical structure JSON file.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        np.ndarray: Binary mask (uint8) of shape (height, width).
    """
    cache_dir = Config.CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{image_id}_tissue_mask.npy")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            mask = np.load(cache_path)
            # Verify shape matches current request (robustness check)
            if mask.shape == (height, width):
                return mask
        except Exception:
            pass  # Fallback to compute if load fails

    # 2. Compute from scratch
    mask = np.zeros((height, width), dtype=np.uint8)
    valid_mask_found = False

    # Fail-Open: Check if file exists
    full_json_path = (
        os.path.join(Config.INPUT_DIR, anatomical_json_path)
        if anatomical_json_path
        else ""
    )

    if anatomical_json_path and os.path.exists(full_json_path):
        try:
            with open(full_json_path, "r") as f:
                annotations = json.load(f)

            # Helper to parse coordinates
            def get_coords(geometry):
                if geometry["type"] == "Polygon":
                    return geometry["coordinates"]
                elif geometry["type"] == "MultiPolygon":
                    coords = []
                    for p in geometry["coordinates"]:
                        coords.extend(p)
                    return coords
                return []

            for ann in annotations:
                # Check for relevant tissue types
                name = (
                    ann.get("properties", {}).get("classification", {}).get("name", "")
                )
                if name in ["Cortex", "Medulla"]:
                    geom = ann.get("geometry", {})
                    coords_list = get_coords(geom)

                    for coords in coords_list:
                        # cv2.fillPoly expects integer coordinates
                        # JSON coords are typically [[x, y], [x, y], ...]
                        pts = np.array(coords, dtype=np.int32)
                        cv2.fillPoly(mask, [pts], 1)
                        valid_mask_found = True

        except Exception as e:
            # If JSON parsing fails, we treat it as no mask found -> Fail Open
            valid_mask_found = False

    # Fail-Open Logic: If no valid mask was generated or file missing, use full image
    if not valid_mask_found or np.sum(mask) == 0:
        if Config.FAIL_OPEN_ROI:
            mask = np.ones((height, width), dtype=np.uint8)
        # If FAIL_OPEN_ROI is False, we return the empty mask (zeros)

    # 3. Save to cache
    try:
        np.save(cache_path, mask)
    except Exception:
        pass  # Non-critical failure

    return mask
