import os
import random
import logging
import numpy as np
import torch
import cv2
from library.config import Config


def get_logger(name: str = "train"):
    """
    Creates and returns a logger with the specified name.
    Ensures handlers are not duplicated if the logger is retrieved multiple times.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def seed_everything(seed: int = 42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Configures CUDA for deterministic execution.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_dicom_image(
    path: str, target_size: tuple = (Config.IMG_SIZE, Config.IMG_SIZE)
) -> np.ndarray:
    """
    Reads a DICOM image from the specified path with a robust fallback mechanism.

    1. Attempts to read using OpenCV.
    2. If OpenCV fails, falls back to raw binary reading from the end of the file,
       assuming the file contains a square 16-bit image.
    3. Converts to float32 and resizes to target_size.

    Args:
        path (str): Path to the .dcm file.
        target_size (tuple): Desired output size (width, height).

    Returns:
        np.ndarray: Processed image as float32 array.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image file not found: {path}")

    # Attempt 1: Standard OpenCV Read
    # IMREAD_UNCHANGED preserves 16-bit depth if present
    try:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    except Exception:
        img = None

    # Attempt 2: Fallback Raw Binary Tail-Read
    # This addresses cases where the DICOM header is unreadable by OpenCV
    if img is None:
        try:
            file_size = os.path.getsize(path)
            # Heuristic: Assume 16-bit depth (2 bytes per pixel)
            # We look for the largest square image that fits in the file
            # File Size >= Header + (H * W * 2)
            max_pixels = file_size // 2
            dim = int(np.sqrt(max_pixels))

            if dim > 0:
                num_bytes = dim * dim * 2
                with open(path, "rb") as f:
                    # Seek to the end minus the image data size
                    f.seek(-num_bytes, os.SEEK_END)
                    data = f.read(num_bytes)

                # Load as uint16 and reshape
                img = np.frombuffer(data, dtype=np.uint16).reshape((dim, dim))
            else:
                # File too small to contain meaningful data
                img = np.zeros(target_size, dtype=np.float32)
        except Exception:
            # If fallback fails, return zeros to maintain pipeline stability
            img = np.zeros(target_size, dtype=np.float32)

    # Final safety check
    if img is None:
        img = np.zeros(target_size, dtype=np.float32)

    # Convert to float32 for model consumption
    img = img.astype(np.float32)

    # Resize to target dimensions
    # cv2.resize expects (width, height)
    current_h, current_w = img.shape[:2]
    target_w, target_h = target_size

    if current_h != target_h or current_w != target_w:
        img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)

    return img
