import os
import random
import re
import numpy as np
import torch
import pydicom
import cv2
from library.config import Config


def seed_everything(seed=None):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int, optional): The seed value to set. If None, uses Config.SEED.
    """
    if seed is None:
        seed = Config.SEED

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _extract_number(filename):
    """
    Helper to extract the integer number from a filename for sorting.
    Expected format: 'Image-123.dcm' or similar containing digits.
    """
    match = re.search(r"(\d+)", filename)
    if match:
        return int(match.group(1))
    return 0


def read_dicom_file(path):
    """
    Attempts to read a single DICOM file path and returns the pixel array.
    Tries pydicom first, then cv2 as a fallback.
    """
    # Attempt 1: pydicom
    try:
        ds = pydicom.dcmread(path)
        return ds.pixel_array
    except Exception:
        pass

    # Attempt 2: OpenCV
    try:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            return img
    except Exception:
        pass

    return None


def load_middle_slice(directory_path):
    """
    Loads the middle slice from a directory of DICOM files.

    1. Lists all .dcm files in the directory.
    2. Sorts them numerically by the index in the filename.
    3. Selects the median file.
    4. Reads and returns the pixel data.

    Args:
        directory_path (str): Path to the directory containing .dcm files.

    Returns:
        np.ndarray: The pixel array of the middle slice.
                    Returns a zero array of shape (Config.IMG_SIZE, Config.IMG_SIZE)
                    if the directory is empty or reading fails.
    """
    # Default fallback shape
    fallback_shape = (Config.IMG_SIZE, Config.IMG_SIZE)

    if not os.path.exists(directory_path):
        return np.zeros(fallback_shape, dtype=np.uint8)

    files = [f for f in os.listdir(directory_path) if f.endswith(".dcm")]

    if not files:
        return np.zeros(fallback_shape, dtype=np.uint8)

    # Sort files numerically to ensure correct spatial ordering
    files.sort(key=_extract_number)

    # Select middle slice
    middle_index = len(files) // 2
    middle_file = files[middle_index]
    full_path = os.path.join(directory_path, middle_file)

    img = read_dicom_file(full_path)

    if img is None:
        return np.zeros(fallback_shape, dtype=np.uint8)

    return img
