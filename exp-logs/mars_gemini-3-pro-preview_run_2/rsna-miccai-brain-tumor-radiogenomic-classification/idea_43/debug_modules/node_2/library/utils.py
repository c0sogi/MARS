import os
import random
import numpy as np
import cv2
import torch
import rasterio
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_dicom_image(path):
    """
    Robustly loads a DICOM image.
    Strategy:
    1. Attempt loading with Rasterio (GDAL).
    2. Attempt direct loading with OpenCV.
    3. Fallback to reading raw binary bytes and decoding (tail-read strategy).

    Args:
        path (str): Path to the .dcm file.

    Returns:
        np.ndarray: The image array, or None if loading fails.
    """
    # Attempt 1: Rasterio (GDAL)
    # Cite debug_lesson_15: Never Silently Swallow Exceptions in Fallback Chains
    with rasterio.open(path) as src:
        img = src.read()
        # Rasterio reads as (C, H, W), convert to (H, W) or (H, W, C)
        if img.shape[0] == 1:
            img = img[0]
        else:
            img = np.transpose(img, (1, 2, 0))
        return img

    # Attempt 2: Standard OpenCV read
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    # Attempt 3: Binary Tail-Read Fallback
    if img is None:
        try:
            with open(path, "rb") as f:
                # Read the entire file buffer
                buffer = np.frombuffer(f.read(), dtype=np.uint8)
            # Decode from buffer
            img = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
        except Exception:
            # If all fail, return None to be handled by the caller (Circuit Breaker)
            return None

    return img


def resize_image(img, size=Config.IMG_SIZE):
    """
    Resizes an image to the specified square dimensions using Area Interpolation.
    Area interpolation is preferred for downsampling to avoid aliasing.

    Args:
        img (np.ndarray): Input image.
        size (int): Target width and height.

    Returns:
        np.ndarray: Resized image.
    """
    if img is None:
        return None

    # cv2.resize expects (width, height)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def normalize_image(img):
    """
    Applies independent Min-Max scaling to the image, mapping values to [0, 1].
    Converts data to float32.

    Args:
        img (np.ndarray): Input image.

    Returns:
        np.ndarray: Normalized image in float32.
    """
    if img is None:
        return None

    img = img.astype(np.float32)

    min_val = np.min(img)
    max_val = np.max(img)

    if max_val > min_val:
        img = (img - min_val) / (max_val - min_val)
    else:
        # If image is constant (e.g., all zeros), return zeros
        img = img - min_val

    return img
