import os
import json
import random
import numpy as np
import cv2
import torch


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def rle_encode(img):
    """
    Encodes a binary mask into Run-Length Encoding (RLE) format.
    The mask is flattened in column-major order (Fortran-style) before encoding,
    consistent with the competition format.

    Args:
        img (np.ndarray): Binary mask (0s and 1s).

    Returns:
        str: RLE string (start length start length ...).
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
        np.ndarray: Binary mask of the specified shape.
    """
    if not isinstance(mask_rle, str) or mask_rle == "nan":
        return np.zeros(shape, dtype=np.uint8)

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order="F")


def polygons_to_mask(json_path, shape, label_name="Cortex"):
    """
    Parses an anatomical structure JSON file and rasterizes polygons of a specific class
    into a binary mask. This is used to create the anatomical prior channel.

    Args:
        json_path (str): Path to the JSON file containing anatomical structures.
        shape (tuple): Output mask shape (height, width).
        label_name (str or list): The classification name(s) to filter (e.g., 'Cortex').

    Returns:
        np.ndarray: Binary mask where pixels inside the specified polygons are 1.
    """
    mask = np.zeros(shape, dtype=np.uint8)

    if not os.path.exists(json_path):
        return mask

    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except Exception:
        return mask

    target_labels = [label_name] if isinstance(label_name, str) else label_name
    polygons = []

    for feature in data:
        # Safely access properties to verify classification
        properties = feature.get("properties", {})
        classification = properties.get("classification", {})
        name = classification.get("name", "")

        if name in target_labels:
            geometry = feature.get("geometry", {})
            if geometry.get("type") == "Polygon":
                coordinates = geometry.get("coordinates", [])
                # Coordinates are typically a list of rings (exterior + holes)
                # We rasterize all rings as filled polygons
                for ring in coordinates:
                    # Convert to numpy array of shape (N, 2) and ensure int32 for cv2
                    pts = np.array(ring, dtype=np.int32)
                    polygons.append(pts)

    if polygons:
        # cv2.fillPoly expects a list of arrays
        cv2.fillPoly(mask, polygons, 1)

    return mask
