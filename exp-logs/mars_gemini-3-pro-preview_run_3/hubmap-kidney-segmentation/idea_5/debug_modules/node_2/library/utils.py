import os
import json
import random
import numpy as np
import cv2
import torch


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.
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
    Encodes a binary mask using Run-Length Encoding (RLE).

    Args:
        img (np.ndarray): Binary mask image (0s and 1s), shape (H, W).

    Returns:
        str: RLE string "start length start length ..."
    """
    # The competition specifies pixels are numbered from top to bottom, then left to right.
    # This corresponds to Fortran-style flattening.
    pixels = img.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def create_tissue_mask(
    anatomical_json_path,
    height,
    width,
    load_cached_data=True,
    cache_dir="./working/idea_5/",
):
    """
    Generates a binary tissue mask (Cortex + Medulla) from an anatomical structure JSON file.
    Implements caching to speed up repeated access.

    Args:
        anatomical_json_path (str): Path to the JSON file containing anatomical structures.
        height (int): Height of the image/mask.
        width (int): Width of the image/mask.
        load_cached_data (bool): If True, attempts to load from cache.
        cache_dir (str): Directory to store cached .npy mask files.

    Returns:
        np.ndarray: Binary mask (uint8) where 1 indicates tissue (Cortex/Medulla) and 0 indicates background.
    """
    # Ensure cache directory exists
    os.makedirs(cache_dir, exist_ok=True)

    # Construct a unique cache filename
    # We use the JSON filename and dimensions to ensure uniqueness
    base_name = os.path.splitext(os.path.basename(anatomical_json_path))[0]
    cache_filename = f"{base_name}_{height}x{width}_tissue_mask.npy"
    cache_path = os.path.join(cache_dir, cache_filename)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            mask = np.load(cache_path)
            # Verify shape matches to avoid stale cache issues
            if mask.shape == (height, width):
                return mask
        except Exception:
            # If load fails, proceed to compute
            pass

    # 2. Compute from scratch
    mask = np.zeros((height, width), dtype=np.uint8)

    # Handle cases where the JSON path might be NaN or file doesn't exist (e.g. test set sometimes)
    if not isinstance(anatomical_json_path, str) or not os.path.exists(
        anatomical_json_path
    ):
        # If no annotation is available, we might assume the whole image is valid or none.
        # However, usually there is an anatomical file. If missing, return empty mask or handle upstream.
        # Here we return empty mask to be safe.
        pass
    else:
        with open(anatomical_json_path, "r") as f:
            data = json.load(f)

        for feature in data:
            # Check if the feature is Cortex or Medulla
            classification = feature.get("properties", {}).get("classification", {})
            name = classification.get("name", "")

            if name in ["Cortex", "Medulla"]:
                geometry = feature.get("geometry", {})
                coordinates = geometry.get("coordinates", [])

                # Coordinates are typically a list of lists of points (polygons)
                # cv2.fillPoly expects a list of numpy arrays of points
                for polygon in coordinates:
                    pts = np.array(polygon, dtype=np.int32)
                    # Reshape to (N, 1, 2) required by OpenCV
                    pts = pts.reshape((-1, 1, 2))
                    cv2.fillPoly(mask, [pts], 1)

    # 3. Save to cache
    try:
        np.save(cache_path, mask)
    except Exception:
        pass

    return mask
