import os
import random
import numpy as np
import torch
import cv2

# Attempt to import pydicom safely; if not installed, fallback to None
try:
    import pydicom
except ImportError:
    pydicom = None


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
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


def load_dicom_as_array(path, size=None):
    """
    Reads a DICOM file and converts it to a float32 numpy array.
    Attempts to use pydicom first, then falls back to OpenCV.

    Args:
        path (str): Path to the DICOM file.
        size (tuple or int, optional): Target size (width, height) or (size, size).
                                       If None, original size is kept.

    Returns:
        np.ndarray: The image data as a float32 array. Returns a zero array if loading fails.
    """
    # Determine default size for fallback (defaulting to 224 if not specified,
    # though usually size is passed by the caller)
    if size is None:
        fallback_size = (224, 224)
    elif isinstance(size, int):
        fallback_size = (size, size)
    else:
        fallback_size = size

    if not os.path.exists(path):
        return np.zeros(fallback_size, dtype=np.float32)

    img = None

    # Attempt 1: pydicom (Preferred for raw DICOM data)
    if pydicom is not None:
        try:
            ds = pydicom.dcmread(path)
            img = ds.pixel_array
        except Exception:
            pass

    # Attempt 2: cv2 (Fallback)
    if img is None:
        try:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        except Exception:
            pass

    # Fallback if both fail
    if img is None:
        return np.zeros(fallback_size, dtype=np.float32)

    # Convert to float32
    img = img.astype(np.float32)

    # Resize if requested
    if size is not None:
        if isinstance(size, int):
            target_size = (size, size)
        else:
            target_size = size

        # cv2.resize expects (width, height)
        if img.size > 0:
            img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        else:
            img = np.zeros(target_size, dtype=np.float32)

    return img


def min_max_scale(img):
    """
    Scales the image pixel values to the range [0, 1].
    Handles cases where max == min (constant image) to avoid division by zero.

    Args:
        img (np.ndarray): Input image array.

    Returns:
        np.ndarray: Scaled image array.
    """
    min_val = np.min(img)
    max_val = np.max(img)

    if max_val - min_val > 0:
        return (img - min_val) / (max_val - min_val)
    else:
        # If image is constant (e.g., all zeros), return zeros
        return np.zeros_like(img)
