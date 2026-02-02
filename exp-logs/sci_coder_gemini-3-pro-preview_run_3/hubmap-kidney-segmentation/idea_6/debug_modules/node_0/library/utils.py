import os
import random
import json
import numpy as np
import pandas as pd
import cv2
import torch
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def rle_encode(img):
    """
    Encodes a binary mask into Run-Length Encoding (RLE).
    The mask is flattened in column-major order (Fortran-style) as per competition specs.

    Args:
        img (np.ndarray): Binary mask (0 or 1), shape (H, W).

    Returns:
        str: RLE string "start length start length ...".
    """
    pixels = img.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle, shape):
    """
    Decodes a Run-Length Encoded (RLE) string into a binary mask.

    Args:
        mask_rle (str): RLE string.
        shape (tuple): Target shape (height, width).

    Returns:
        np.ndarray: Binary mask, shape (H, W).
    """
    if pd.isna(mask_rle) or mask_rle == "":
        return np.zeros(shape, dtype=np.uint8)

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
    Generates a binary mask for valid tissue regions (Cortex/Medulla) from an anatomical JSON file.
    Implements caching to avoid re-parsing and rasterizing polygons on every call.

    Args:
        image_id (str): Unique identifier for the image.
        width (int): Width of the image.
        height (int): Height of the image.
        anatomical_json_path (str): Path to the JSON file containing anatomical structures.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        np.ndarray: Binary mask (uint8) where 1 indicates tissue and 0 indicates background.
    """
    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Define cache filename
    cache_filename = f"{image_id}_{width}x{height}_tissue_mask.npy"
    cache_path = os.path.join(Config.CACHE_DIR, cache_filename)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            mask = np.load(cache_path)
            # Basic validation of shape to ensure cache validity
            if mask.shape == (height, width):
                return mask
        except Exception:
            # If load fails, proceed to recompute
            pass

    # 2. Compute from scratch
    mask = np.zeros((height, width), dtype=np.uint8)

    # Resolve path: if relative and not found, check in input dir
    full_json_path = anatomical_json_path
    if not os.path.exists(full_json_path):
        full_json_path = os.path.join(Config.INPUT_DIR, anatomical_json_path)

    if os.path.exists(full_json_path):
        try:
            with open(full_json_path, "r") as f:
                data = json.load(f)

            # Iterate over features (anatomical structures)
            if isinstance(data, list):
                for feature in data:
                    geometry = feature.get("geometry", {})
                    # We include all anatomical structures (Cortex, Medulla) as tissue
                    if geometry.get("type") == "Polygon":
                        coordinates = geometry.get("coordinates", [])
                        for ring in coordinates:
                            # ring is a list of [x, y] points
                            pts = np.array(ring, dtype=np.int32)
                            pts = pts.reshape((-1, 1, 2))
                            cv2.fillPoly(mask, [pts], 1)
        except Exception as e:
            print(f"Error processing JSON for {image_id}: {e}")

    # 3. Save to cache
    try:
        np.save(cache_path, mask)
    except Exception as e:
        print(f"Warning: Failed to save cache for {image_id}: {e}")

    return mask
