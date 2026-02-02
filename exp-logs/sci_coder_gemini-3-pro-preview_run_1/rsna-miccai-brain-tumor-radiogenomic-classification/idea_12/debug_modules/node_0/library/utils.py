import os
import random
import numpy as np
import cv2
import torch
import pydicom
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_and_normalize_dicom(path):
    """
    Reads a DICOM file, resizes it to the configured image size, converts it to float32,
    and applies Min-Max scaling to normalize the pixel values to the range [0, 1].

    Args:
        path (str): The file path to the DICOM image.

    Returns:
        np.ndarray: A 2D numpy array of shape (Config.IMAGE_SIZE, Config.IMAGE_SIZE)
                    with float32 values in [0, 1].
    """
    img = None

    # Attempt 1: Read using pydicom (preferred for DICOM)
    try:
        dcm = pydicom.dcmread(path)
        img = dcm.pixel_array
    except Exception:
        pass

    # Attempt 2: Read using OpenCV if pydicom fails
    if img is None:
        try:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        except Exception:
            pass

    # If reading failed completely, return a zero placeholder
    if img is None:
        return np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE), dtype=np.float32)

    # Resize image to the target size defined in Config
    # Using INTER_LINEAR or INTER_CUBIC for resizing
    try:
        if img.shape[0] != Config.IMAGE_SIZE or img.shape[1] != Config.IMAGE_SIZE:
            img = cv2.resize(
                img,
                (Config.IMAGE_SIZE, Config.IMAGE_SIZE),
                interpolation=cv2.INTER_LINEAR,
            )
    except Exception:
        # Fallback for empty or malformed arrays
        return np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE), dtype=np.float32)

    # Convert to float32 for precision
    img = img.astype(np.float32)

    # Min-Max Normalization
    img_min = img.min()
    img_max = img.max()

    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min)
    else:
        # Avoid division by zero if the image is constant (e.g., all black)
        img = np.zeros_like(img)

    return img
