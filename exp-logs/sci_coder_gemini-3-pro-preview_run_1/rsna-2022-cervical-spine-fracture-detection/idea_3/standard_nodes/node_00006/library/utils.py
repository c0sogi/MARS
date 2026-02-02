import os
import numpy as np
import pydicom
import cv2
import torch
import pandas as pd
from library.config import Config

# -------------------------------------------------------------------------
# DICOM and Image Processing
# -------------------------------------------------------------------------


def load_dicom(path):
    """
    Reads a DICOM file from the specified path.

    Args:
        path (str): Path to the .dcm file.

    Returns:
        pydicom.dataset.FileDataset: The loaded DICOM dataset.
    """
    try:
        ds = pydicom.dcmread(path)
        return ds
    except Exception as e:
        print(f"Error loading DICOM {path}: {e}")
        return None


def convert_to_hu(ds):
    """
    Converts a DICOM dataset's pixel array to Hounsfield Units (HU).

    Args:
        ds (pydicom.dataset.FileDataset): The DICOM dataset.

    Returns:
        np.ndarray: The pixel array in HU.
    """
    try:
        image = ds.pixel_array.astype(np.float32)

        # Apply Rescale Slope and Intercept if present
        slope = getattr(ds, "RescaleSlope", 1.0)
        intercept = getattr(ds, "RescaleIntercept", 0.0)

        if slope != 1.0:
            image = slope * image
        if intercept != 0.0:
            image += intercept

        return image
    except Exception as e:
        # Fallback for compressed images or missing pixel data if libraries missing
        # print(f"Error converting to HU: {e}")
        return np.zeros((512, 512), dtype=np.float32)


def apply_bone_window(image, center=Config.WINDOW_CENTER, width=Config.WINDOW_WIDTH):
    """
    Applies a windowing function to the image (typically for bone visualization).
    Normalizes the output to the range [0, 1].

    Args:
        image (np.ndarray): Input image in HU.
        center (float): Window center.
        width (float): Window width.

    Returns:
        np.ndarray: Windowed and normalized image (float32, 0.0 to 1.0).
    """
    min_value = center - width // 2
    max_value = center + width // 2

    windowed = np.clip(image, min_value, max_value)
    # Normalize to [0, 1]
    if max_value > min_value:
        windowed = (windowed - min_value) / (max_value - min_value)
    else:
        windowed = windowed - min_value  # Fallback

    return windowed.astype(np.float32)


def process_dicom(path, resize_to=None):
    """
    Pipeline to load, convert to HU, and apply bone window.

    Args:
        path (str): Path to DICOM file.
        resize_to (tuple, optional): (height, width) to resize.

    Returns:
        np.ndarray: Processed image.
    """
    ds = load_dicom(path)
    if ds is None:
        return np.zeros((512, 512), dtype=np.float32)

    image = convert_to_hu(ds)
    image = apply_bone_window(image)

    if resize_to is not None:
        image = cv2.resize(image, (resize_to[1], resize_to[0]))

    return image


def crop_image(image, center_yx, crop_size_hw):
    """
    Crops an image around a center point with padding if necessary.

    Args:
        image (np.ndarray): Input image (H, W).
        center_yx (tuple): (y, x) coordinates of the center.
        crop_size_hw (tuple): (height, width) of the crop.

    Returns:
        np.ndarray: Cropped image.
    """
    h, w = image.shape[:2]
    cy, cx = center_yx
    ch, cw = crop_size_hw

    # Calculate boundaries
    y1 = int(cy - ch // 2)
    y2 = int(cy + ch // 2)
    x1 = int(cx - cw // 2)
    x2 = int(cx + cw // 2)

    # Calculate padding if out of bounds
    pad_top = max(0, -y1)
    pad_bottom = max(0, y2 - h)
    pad_left = max(0, -x1)
    pad_right = max(0, x2 - w)

    # Adjust crop coordinates for the original image
    y1 = max(0, y1)
    y2 = min(h, y2)
    x1 = max(0, x1)
    x2 = min(w, x2)

    # Crop
    cropped = image[y1:y2, x1:x2]

    # Pad
    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
        cropped = np.pad(
            cropped,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=0,
        )

    # Ensure exact size (in case of rounding issues)
    if cropped.shape[0] != ch or cropped.shape[1] != cw:
        cropped = cv2.resize(cropped, (cw, ch))

    return cropped


# Caching and I/O
# -------------------------------------------------------------------------


def save_cache(data, filename, use_parquet=False):
    """
    Saves data to the cache directory.

    Args:
        data: The data to save (DataFrame or object/array).
        filename (str): Name of the file.
        use_parquet (bool): If True, treats data as DataFrame and saves as parquet.
                            Otherwise uses np.save.
    """
    cache_path = os.path.join(Config.CACHE_DIR, filename)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    try:
        if use_parquet:
            if isinstance(data, pd.DataFrame):
                data.to_parquet(cache_path)
            else:
                print("Error: use_parquet=True but data is not DataFrame.")
        else:
            np.save(cache_path, data)
    except Exception as e:
        print(f"Failed to save cache {filename}: {e}")


def load_cache(filename, use_parquet=False):
    """
    Loads data from the cache directory if it exists.

    Args:
        filename (str): Name of the file.
        use_parquet (bool): If True, loads as parquet.

    Returns:
        The loaded data, or None if not found.
    """
    cache_path = os.path.join(Config.CACHE_DIR, filename)

    if not os.path.exists(cache_path):
        return None

    try:
        if use_parquet:
            return pd.read_parquet(cache_path)
        else:
            return np.load(cache_path, allow_pickle=True)
    except Exception as e:
        print(f"Failed to load cache {filename}: {e}")
        return None


# -------------------------------------------------------------------------
# Metrics
# -------------------------------------------------------------------------


def calculate_weighted_loss(y_true, y_pred):
    """
    Calculates the competition weighted log loss.

    Args:
        y_true (np.ndarray): Binary labels (N, 8).
        y_pred (np.ndarray): Probabilities (N, 8).

    Returns:
        float: The weighted loss.
    """
    # Weights: patient_overall (col 0) = 7, others = 1
    # Assuming order: [patient_overall, C1, C2, C3, C4, C5, C6, C7]
    # Or based on prompt: "The any label is weighted more highly"

    # Define weights
    weights = np.ones(8)
    weights[0] = Config.WEIGHT_PATIENT_OVERALL  # patient_overall
    weights[1:] = Config.WEIGHT_VERTEBRAE  # C1-C7

    # Clip predictions to avoid log(0)
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    # Calculate Log Loss per element
    # L = -w * [y * log(p) + (1-y) * log(1-p)]
    loss = -weights * (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    # Average across all samples and columns
    return np.mean(loss)
