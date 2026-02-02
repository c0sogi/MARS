import os
import json
import cv2
import numpy as np
import torch
from library.config import Config


def rle_encode(img):
    """
    Encodes a binary mask into Run-Length Encoding (RLE).

    The pixels are numbered from top to bottom, then left to right (Column-Major).

    Args:
        img (np.ndarray): Binary mask image of shape (Height, Width).
                          Values should be 0 or 1.

    Returns:
        str: RLE encoded string 'start length start length ...'.
    """
    # Flatten in column-major order (Fortran-style)
    pixels = img.flatten(order="F")

    # Pad with zeros to detect runs at the start/end
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where values change
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # Calculate lengths (end - start)
    runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle, shape):
    """
    Decodes a Run-Length Encoded string into a binary mask.

    Args:
        mask_rle (str): RLE encoded string.
        shape (tuple): Target shape (Height, Width).

    Returns:
        np.ndarray: Binary mask of shape (Height, Width).
    """
    if not isinstance(mask_rle, str) or not mask_rle:
        return np.zeros(shape, dtype=np.uint8)

    s = mask_rle.split()
    # Extract starts and lengths
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]

    # Adjust 1-based indexing to 0-based
    starts -= 1
    ends = starts + lengths

    # Create flattened mask
    total_pixels = shape[0] * shape[1]
    img = np.zeros(total_pixels, dtype=np.uint8)

    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    # Reshape back to image dimensions (Column-Major)
    return img.reshape(shape, order="F")


def polygons_to_mask(polygons, shape):
    """
    Converts a list of polygon coordinates into a binary mask.

    Args:
        polygons (list): List of polygon rings. Each ring is a list of [x, y] coordinates.
        shape (tuple): Target shape (Height, Width).

    Returns:
        np.ndarray: Binary mask where polygon regions are 1 and background is 0.
    """
    mask = np.zeros(shape, dtype=np.uint8)
    if not polygons:
        return mask

    # Convert list of lists to list of numpy arrays for cv2
    # polygons structure is expected to be [ [[x,y], [x,y], ...], ... ]
    formatted_polys = [np.array(p, dtype=np.int32) for p in polygons]

    # Fill polygons with 1
    cv2.fillPoly(mask, formatted_polys, 1)

    return mask


def get_tissue_mask(
    json_path, shape, valid_classes=["Cortex", "Medulla"], load_cached_data=True
):
    """
    Generates a tissue mask from an anatomical structure JSON file with caching.

    Args:
        json_path (str): Path to the anatomical structure JSON file.
        shape (tuple): (Height, Width) of the corresponding image.
        valid_classes (list): List of classification names to include (e.g., 'Cortex').
        load_cached_data (bool): If True, attempts to load from cache before computing.

    Returns:
        np.ndarray: Binary mask of the tissue regions.
    """
    # Define cache directory and ensure it exists
    cache_dir = os.path.join(Config.ARTIFACT_DIR, "tissue_masks_cache")
    os.makedirs(cache_dir, exist_ok=True)

    # Construct a unique cache filename
    base_name = os.path.splitext(os.path.basename(json_path))[0]
    classes_str = "_".join(sorted(valid_classes))
    cache_filename = f"{base_name}_{shape[0]}x{shape[1]}_{classes_str}.npy"
    cache_path = os.path.join(cache_dir, cache_filename)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            return np.load(cache_path)
        except Exception:
            # Fallback to re-computing if load fails
            pass

    # 2. Compute from scratch
    mask = np.zeros(shape, dtype=np.uint8)

    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            data = json.load(f)

        polygons = []
        for feature in data:
            # Extract classification name
            props = feature.get("properties", {})
            classification = props.get("classification", {})
            name = classification.get("name")

            # Filter by class
            if name in valid_classes:
                geometry = feature.get("geometry", {})
                # GeoJSON Polygon coordinates are a list of rings (exterior, interior...)
                coordinates = geometry.get("coordinates", [])
                polygons.extend(coordinates)

        if polygons:
            mask = polygons_to_mask(polygons, shape)

    # 3. Save to cache
    np.save(cache_path, mask)

    return mask


def dice_coef(y_true, y_pred, smooth=1.0):
    """
    Calculates the Dice Coefficient for PyTorch tensors.

    Args:
        y_true (torch.Tensor): Ground truth binary mask.
        y_pred (torch.Tensor): Predicted probabilities or binary mask.
        smooth (float): Smoothing factor.

    Returns:
        torch.Tensor: Dice coefficient score.
    """
    # Flatten the tensors
    y_true_f = y_true.view(-1)
    y_pred_f = y_pred.view(-1)

    intersection = (y_true_f * y_pred_f).sum()
    return (2.0 * intersection + smooth) / (y_true_f.sum() + y_pred_f.sum() + smooth)
