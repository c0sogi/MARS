import os
import re
import random
import numpy as np
import torch
import cv2
import pydicom
from library.config import IMAGE_SIZE


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_sorted_file_paths(folder_path):
    """
    Retrieves and sorts DICOM file paths from a directory based on the instance number
    embedded in the filename (e.g., 'Image-10.dcm').

    Args:
        folder_path (str): Path to the directory containing DICOM files.

    Returns:
        list: A sorted list of full file paths.
    """
    if not os.path.exists(folder_path):
        return []

    files = [f for f in os.listdir(folder_path) if f.endswith(".dcm")]

    def extract_number(filename):
        # Matches 'Image-123.dcm' and extracts 123
        match = re.search(r"Image-(\d+)\.dcm", filename)
        if match:
            return int(match.group(1))
        # Fallback if format is different, try finding any number
        numbers = re.findall(r"\d+", filename)
        if numbers:
            return int(numbers[-1])
        return 0

    # Sort files based on the extracted number
    sorted_files = sorted(files, key=extract_number)

    return [os.path.join(folder_path, f) for f in sorted_files]


def load_dicom_slice(path, target_size=IMAGE_SIZE):
    """
    Reads a DICOM file, resizes it, and normalizes pixel values to [0, 1].
    Attempts to use pydicom first, then falls back to OpenCV.

    Args:
        path (str): Path to the DICOM file.
        target_size (int): The spatial dimension to resize the image to (target_size x target_size).

    Returns:
        np.ndarray: A 2D numpy array of shape (target_size, target_size) with float32 values in [0, 1].
                    Returns a zero array if reading fails.
    """
    img = None

    # Attempt 1: pydicom
    try:
        ds = pydicom.dcmread(path)
        img = ds.pixel_array
    except Exception:
        pass

    # Attempt 2: OpenCV
    if img is None:
        try:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        except Exception:
            pass

    # Fallback: Return black image if read fails
    if img is None:
        return np.zeros((target_size, target_size), dtype=np.float32)

    # Resize
    if img.shape[0] != target_size or img.shape[1] != target_size:
        img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)

    # Normalize to [0, 1]
    img = img.astype(np.float32)
    min_val = np.min(img)
    max_val = np.max(img)

    if max_val > min_val:
        img = (img - min_val) / (max_val - min_val)
    else:
        img = np.zeros_like(img)  # Avoid division by zero for constant images

    return img
