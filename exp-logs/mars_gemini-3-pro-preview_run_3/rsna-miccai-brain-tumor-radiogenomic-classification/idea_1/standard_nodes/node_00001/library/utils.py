import os
import random
import numpy as np
import torch
import pydicom
import cv2
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
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


def load_dicom(rel_path):
    """
    Reads a DICOM file from the input directory.

    Args:
        rel_path (str): Relative path to the DICOM file (e.g., 'train/00000/FLAIR/Image-1.dcm').

    Returns:
        np.ndarray: The pixel array of the DICOM image. Returns None if loading fails.
    """
    # Construct full path using Config.INPUT_DIR
    full_path = os.path.join(Config.INPUT_DIR, rel_path)

    if not os.path.exists(full_path):
        return None

    try:
        dcm = pydicom.dcmread(full_path)
        img = dcm.pixel_array
        return img
    except Exception:
        return None


def normalize_pixels(img):
    """
    Normalizes pixel intensity values to the range [0, 1] using Min-Max scaling.

    Args:
        img (np.ndarray): Input image array.

    Returns:
        np.ndarray: Normalized image array. Returns None if input is None.
    """
    if img is None:
        return None

    img = img.astype(np.float32)
    min_val = np.min(img)
    max_val = np.max(img)

    # Avoid division by zero
    if max_val > min_val:
        img = (img - min_val) / (max_val - min_val)
    else:
        img = np.zeros_like(img)

    return img


def resize_image(img, size=Config.IMAGE_SIZE):
    """
    Resizes an image to the specified square dimensions.

    Args:
        img (np.ndarray): Input image.
        size (int): Target width and height. Defaults to Config.IMAGE_SIZE.

    Returns:
        np.ndarray: Resized image. Returns None if input is None.
    """
    if img is None:
        return None

    # cv2.resize expects (width, height)
    try:
        resized = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        return resized
    except Exception:
        return None


def select_middle_indices(num_files, num_slices=Config.NUM_SLICES):
    """
    Calculates the indices of the middle slices to extract from a list of files.

    Args:
        num_files (int): Total number of files available.
        num_slices (int): Number of slices to extract. Defaults to Config.NUM_SLICES.

    Returns:
        List[int]: List of indices to select.
    """
    if num_files == 0:
        return []

    if num_files < num_slices:
        # If fewer files than requested, return all available indices
        return list(range(num_files))

    # Calculate start index to center the window
    start = (num_files - num_slices) // 2
    return list(range(start, start + num_slices))
