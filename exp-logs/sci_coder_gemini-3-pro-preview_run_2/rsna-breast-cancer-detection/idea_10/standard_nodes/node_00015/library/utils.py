import os
import random
import numpy as np
import torch
import cv2
from library.config import Config


def seed_everything(seed: int = Config.SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
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


def get_device() -> torch.device:
    """
    Returns the PyTorch device specified in the configuration.

    Returns:
        torch.device: The device (cpu or cuda).
    """
    return torch.device(Config.DEVICE)


def load_image(path: str, size: tuple = Config.IMG_SIZE) -> np.ndarray:
    """
    Loads an image from the specified path using OpenCV.
    Handles DICOM files by treating them as image files (fallback mechanism).
    Normalizes 16-bit/high-bit depth images to 8-bit [0, 255].
    Resizes the image to the specified dimensions.

    Args:
        path (str): Path to the image file.
        size (tuple): Target size as (Height, Width). Defaults to Config.IMG_SIZE.

    Returns:
        np.ndarray: The loaded and processed image as a uint8 array.
    """
    # Check if file exists
    if not os.path.exists(path):
        # Return black image if file missing
        return np.zeros(size, dtype=np.uint8)

    # Attempt to load using OpenCV
    # IMREAD_UNCHANGED is crucial to preserve original bit depth (e.g., 16-bit)
    try:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    except Exception:
        img = None

    # Handle load failure
    if img is None:
        return np.zeros(size, dtype=np.uint8)

    # Convert to Grayscale if loaded as RGB/BGR (though DICOM is usually single channel)
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Normalize to 8-bit [0, 255]
    # This ensures compatibility with CLAHE and standard CNN input ranges
    if img.dtype != np.uint8:
        img_min = img.min()
        img_max = img.max()

        if img_max > img_min:
            # Min-Max scaling to 0-1, then scale to 255
            img = (img - img_min) / (img_max - img_min)
            img = (img * 255.0).astype(np.uint8)
        else:
            # If image is constant (min == max), return zero array
            img = np.zeros_like(img, dtype=np.uint8)

    # Resize to target dimensions
    # Config.IMG_SIZE is (Height, Width), cv2.resize expects (Width, Height)
    if size is not None:
        target_h, target_w = size
        if img.shape[:2] != (target_h, target_w):
            img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    return img
