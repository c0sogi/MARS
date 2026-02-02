import os
import json
import numpy as np
import cv2
from library.config import Config


def rle_encode(img):
    """
    Run-length encoding for a binary mask.

    Args:
        img (np.ndarray): Binary mask of shape (height, width), where 1 indicates the object.

    Returns:
        str: Space-separated run-length encoding.
    """
    # Flatten column-wise
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle, shape):
    """
    Decodes a run-length encoded string into a binary mask.

    Args:
        mask_rle (str): RLE string.
        shape (tuple): Target shape (height, width).

    Returns:
        np.ndarray: Binary mask of shape (height, width).
    """
    if not mask_rle or str(mask_rle) == "nan":
        return np.zeros(shape, dtype=np.uint8)

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order="F")


def get_tissue_mask_from_json(image_id, json_path, image_shape, load_cached_data=True):
    """
    Generates a binary mask for valid tissue regions (Cortex/Medulla) from an anatomical structure JSON.
    Implements caching to avoid re-parsing and re-drawing polygons.

    Args:
        image_id (str): Unique identifier for the image.
        json_path (str): Path to the anatomical structure JSON file.
        image_shape (tuple): Shape of the image (height, width).
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        np.ndarray: Binary mask (uint8) where 1 indicates tissue, 0 indicates background.
    """
    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Define cache path
    cache_filename = f"{image_id}_tissue_mask.npy"
    cache_path = os.path.join(Config.CACHE_DIR, cache_filename)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            mask = np.load(cache_path)
            # Verify shape matches (simple integrity check)
            if mask.shape == image_shape:
                return mask
        except Exception:
            # If load fails, proceed to compute
            pass

    # 2. Compute from scratch
    mask = np.zeros(image_shape, dtype=np.uint8)

    # Handle case where JSON path might be invalid or missing
    full_json_path = os.path.join(Config.INPUT_ROOT, json_path)
    if not os.path.exists(full_json_path):
        # If no anatomical structure file, assume whole image is valid or return empty?
        # Usually, if missing, we might assume the whole image is valid tissue,
        # but to be safe and consistent with the task of using anatomical structures,
        # we return zeros or handle it. Here we return zeros if file missing.
        # However, based on metadata, paths should exist.
        print(
            f"Warning: Anatomical JSON not found at {full_json_path}. Returning empty mask."
        )
        # Save empty mask to cache to avoid repeated checks
        np.save(cache_path, mask)
        return mask

    try:
        with open(full_json_path, "r") as f:
            annotations = json.load(f)

        # Anatomical structures to include
        valid_structures = {"Cortex", "Medulla"}

        for ann in annotations:
            # Check classification
            properties = ann.get("properties", {})
            classification = properties.get("classification", {})
            name = classification.get("name", "")

            if name in valid_structures:
                geometry = ann.get("geometry", {})
                coordinates = geometry.get("coordinates", [])

                # Coordinates are usually a list of lists of points: [[[x, y], ...]]
                # cv2.fillPoly expects a list of numpy arrays of points
                for polygon in coordinates:
                    pts = np.array(polygon, dtype=np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    cv2.fillPoly(mask, [pts], 1)

    except Exception as e:
        print(f"Error parsing JSON for {image_id}: {e}")
        # Return whatever was computed so far (likely empty or partial)

    # 3. Save to cache
    np.save(cache_path, mask)

    return mask
