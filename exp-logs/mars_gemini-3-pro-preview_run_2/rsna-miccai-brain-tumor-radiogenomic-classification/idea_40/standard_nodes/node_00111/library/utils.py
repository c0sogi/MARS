import os
import sys
import random
import logging
import numpy as np
import torch
import cv2
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Ensures deterministic behavior for model training and data processing.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(name=__name__):
    """
    Returns a configured logger with standard formatting.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def read_dicom_robust(file_path, target_size=Config.IMG_SIZE):
    """
    Robust DICOM reader implementing the 'Robust Loading' strategy.

    1. Attempts to load using OpenCV (cv2.imread).
    2. Falls back to 'Raw Binary Tail-Read' if decoding fails, checking for
       standard square MRI dimensions at the end of the file.
    3. Resizes to target_size using Area Interpolation to suppress noise.
    4. Returns a float32 numpy array.

    Args:
        file_path (str): Path to the DICOM file.
        target_size (int): Desired spatial dimension (H=W).

    Returns:
        np.ndarray: The processed image as a float32 array, or a zero array on failure.
    """
    if not os.path.exists(file_path):
        return np.zeros((target_size, target_size), dtype=np.float32)

    img = None

    # Attempt 1: Standard OpenCV Loading
    try:
        # IMREAD_UNCHANGED is crucial to preserve 16-bit depth of MRI scans
        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    except Exception:
        img = None

    # Attempt 2: Raw Binary Tail-Read (Fallback)
    # This handles cases where DICOM headers are corrupt or incompatible with OpenCV,
    # but the pixel data at the end of the file is intact.
    if img is None:
        try:
            file_size = os.path.getsize(file_path)
            # Standard MRI dimensions (uint16 = 2 bytes per pixel)
            # 512x512 -> 524,288 bytes
            # 256x256 -> 131,072 bytes
            # 128x128 -> 32,768 bytes
            possible_dims = [512, 256, 128]

            with open(file_path, "rb") as f:
                data = f.read()

            for dim in possible_dims:
                expected_bytes = dim * dim * 2
                if file_size >= expected_bytes:
                    # Assume pixel data is located at the very end of the file
                    raw_bytes = data[-expected_bytes:]
                    arr = np.frombuffer(raw_bytes, dtype=np.uint16)
                    # Verify we have the exact amount of data for this dimension
                    if arr.size == dim * dim:
                        img = arr.reshape((dim, dim))
                        break
        except Exception:
            pass

    # Circuit Breaker: If all methods fail, return a black image
    if img is None:
        return np.zeros((target_size, target_size), dtype=np.float32)

    # Dimensionality Correction
    # Handle cases where OpenCV might interpret the file as RGB
    if img.ndim == 3:
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            # If multi-channel but not RGB, take the first channel
            img = img[:, :, 0]

    # Geometric Normalization
    # Resize to target geometry using Area Interpolation to reduce aliasing/noise
    if img.shape[0] != target_size or img.shape[1] != target_size:
        try:
            img = cv2.resize(
                img, (target_size, target_size), interpolation=cv2.INTER_AREA
            )
        except Exception:
            return np.zeros((target_size, target_size), dtype=np.float32)

    return img.astype(np.float32)
