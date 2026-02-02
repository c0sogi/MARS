import os
import random
import numpy as np
import torch
import cv2
import sys

# Attempt to import pydicom, handle gracefully if not present
try:
    import pydicom
except ImportError:
    pydicom = None

from library.config import Config


def seed_everything(seed: int = 42):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Returns the available device (CUDA or CPU).

    Returns:
        torch.device: The device object.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def log_message(message: str):
    """
    Prints a message to stdout with immediate flushing.

    Args:
        message (str): The message to log.
    """
    print(message)
    sys.stdout.flush()


def load_dicom_image(path: str) -> np.ndarray:
    """
    Reads a DICOM file from the specified path using pydicom or cv2.
    Converts the pixel data to a float32 numpy array.

    Args:
        path (str): The file path to the DICOM image.

    Returns:
        np.ndarray: The image data as a float32 array, or None if reading fails.
    """
    if not os.path.exists(path):
        return None

    img = None

    # Attempt 1: Use pydicom (Preferred for DICOM)
    if pydicom is not None:
        try:
            ds = pydicom.dcmread(path)
            img = ds.pixel_array
        except Exception:
            # Fallthrough to next method
            pass

    # Attempt 2: Use OpenCV (Fallback)
    if img is None:
        try:
            # cv2.IMREAD_UNCHANGED is crucial to preserve depth if possible
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        except Exception:
            pass

    # Verification and Conversion
    if img is not None:
        try:
            # Ensure float32 precision as required by the pipeline
            img = img.astype(np.float32)
            return img
        except Exception:
            return None

    return None
