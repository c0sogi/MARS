import os
import random
import re
import numpy as np
import torch
import cv2
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
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


def get_middle_indices(file_list, depth=3):
    """
    Calculates the indices of the middle slices of a volume given a list of files.
    Returns a list of integer indices corresponding to the middle section.

    Args:
        file_list (list): List of file paths or names.
        depth (int): Number of slices to select.

    Returns:
        list: List of integer indices.
    """
    num_files = len(file_list)
    if num_files == 0:
        return []

    # Calculate geometric center
    mid_idx = num_files // 2

    # Calculate start and end indices
    half_depth = depth // 2
    start_idx = mid_idx - half_depth
    end_idx = start_idx + depth

    # Handle boundary conditions
    if start_idx < 0:
        start_idx = 0
        end_idx = min(depth, num_files)

    if end_idx > num_files:
        end_idx = num_files
        start_idx = max(0, end_idx - depth)

    return list(range(start_idx, end_idx))


def read_dicom_processed(path, img_size=None):
    """
    Reads a DICOM file, resizes it to the target size, and converts it to a float32 array.

    Args:
        path (str): Path to the DICOM file.
        img_size (int, optional): Target spatial dimension (img_size x img_size).
                                  Defaults to Config.IMG_SIZE.

    Returns:
        np.ndarray: Processed image as a float32 array.
    """
    if img_size is None:
        img_size = Config.IMG_SIZE

    image = None

    # Attempt 1: Try using pydicom (Standard for DICOM)
    try:
        import pydicom

        dcm = pydicom.dcmread(path)
        image = dcm.pixel_array
    except (ImportError, Exception):
        pass

    # Attempt 2: Try using OpenCV (Fallback)
    if image is None:
        try:
            image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        except Exception:
            pass

    # Fallback: Return black image if reading fails
    if image is None:
        return np.zeros((img_size, img_size), dtype=np.float32)

    # Ensure image is 2D (remove channel dim if present)
    if image.ndim == 3:
        # If RGB/BGR, convert to grayscale
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            # If just extra dim (H, W, 1), squeeze it
            image = np.squeeze(image)

    # Resize image
    if image.shape[0] != img_size or image.shape[1] != img_size:
        try:
            image = cv2.resize(
                image, (img_size, img_size), interpolation=cv2.INTER_AREA
            )
        except Exception:
            return np.zeros((img_size, img_size), dtype=np.float32)

    # Convert to float32
    return image.astype(np.float32)
