import os
import sys
import random
import logging
import numpy as np
import cv2
import torch
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.manual_seed(seed)


def setup_logger(name="OS-LDE", log_file=None):
    """
    Configures a logger to output to both the console and a file.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Clear existing handlers to prevent duplicate logging
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Stream Handler (Console)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    # File Handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def load_image(path):
    """
    Loads an image from the specified path using OpenCV.
    Ensures the image is loaded in BGR color format.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found at path: {path}")

    # Load as color (3 channels) to ensure consistency for model inputs
    img = cv2.imread(path, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError(f"Failed to decode image at path: {path}")

    return img


def rotate_image(image, angle):
    """
    Rotates an image by a specific angle (in degrees) around its center.

    This function expands the canvas size to ensure the entire rotated image fits
    without cropping any corners. The background is filled with white (255),
    matching the dataset's binary leaf on white background format.

    Args:
        image (np.ndarray): Input image (H, W, C) or (H, W).
        angle (float): Rotation angle in degrees (counter-clockwise).

    Returns:
        np.ndarray: The rotated image with expanded dimensions.
    """
    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    # Get rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Calculate new bounding box dimensions to prevent cropping
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # Adjust the rotation matrix translation to center the image in the new frame
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    # Determine border color (White)
    if len(image.shape) == 3 and image.shape[2] == 3:
        border_value = (255, 255, 255)
    else:
        border_value = 255

    # Perform rotation with white padding
    rotated = cv2.warpAffine(
        image,
        M,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )

    return rotated


def save_numpy(data, filename):
    """
    Saves a numpy array to the working directory defined in Config.
    Ensures the directory structure exists.
    """
    path = Config.get_cache_path(filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, data)


def load_numpy(filename):
    """
    Loads a numpy array from the working directory defined in Config.
    Returns None if the file does not exist.
    """
    path = Config.get_cache_path(filename)
    if os.path.exists(path):
        # We use allow_pickle=True to support object arrays (e.g., string labels)
        # if necessary, though numerical arrays are preferred.
        return np.load(path, allow_pickle=True)
    return None
