import os
import sys
import random
import logging
import numpy as np
import torch
import cv2
import pydicom
from library.config import Config


def seed_everything(seed: int = 42):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(log_file: str):
    """
    Configures the logger to write to a file and stdout.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Remove existing handlers to avoid duplicates during interactive runs
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def load_dicom(
    path: str,
    size: tuple = (224, 224),
    window_level: float = 400,
    window_width: float = 1800,
):
    """
    Loads a DICOM file, converts to Hounsfield Units (HU), applies windowing,
    and resizes the image.

    Args:
        path (str): Path to the .dcm file.
        size (tuple): Target size (height, width).
        window_level (float): Window center (Level).
        window_width (float): Window width.

    Returns:
        np.ndarray: Processed image as uint8 array (0-255).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"DICOM file not found: {path}")

    try:
        dicom = pydicom.dcmread(path)

        # Extract pixel array and convert to float
        img = dicom.pixel_array.astype(np.float32)

        # Convert to Hounsfield Units (HU)
        # Handle potential missing attributes or list types for slope/intercept
        intercept = getattr(dicom, "RescaleIntercept", 0.0)
        slope = getattr(dicom, "RescaleSlope", 1.0)

        if isinstance(slope, (list, tuple)):
            slope = slope[0]
        if isinstance(intercept, (list, tuple)):
            intercept = intercept[0]

        img = img * float(slope) + float(intercept)

        # Apply Windowing
        lower = window_level - (window_width / 2.0)
        upper = window_level + (window_width / 2.0)

        img = np.clip(img, lower, upper)

        # Normalize to 0-1 range
        if window_width > 0:
            img = (img - lower) / window_width
        else:
            img = img - lower

        # Resize
        # cv2.resize expects (width, height)
        if size:
            img = cv2.resize(img, (size[1], size[0]))

        # Convert to uint8 (0-255) for memory efficiency
        img = (img * 255.0).astype(np.uint8)

        return img

    except Exception as e:
        raise RuntimeError(f"Failed to process DICOM {path}: {e}")
