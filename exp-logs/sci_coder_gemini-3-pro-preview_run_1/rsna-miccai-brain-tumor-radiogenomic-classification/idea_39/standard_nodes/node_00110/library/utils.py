import os
import re
import cv2
import numpy as np
import random
import torch
import glob
from library.config import SEED


def seed_everything(seed=SEED):
    """
    Sets the random seed for various libraries to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_dicom_image(path):
    """
    Reads a DICOM file and returns the pixel array.
    Tries pydicom first, then falls back to OpenCV.
    Returns None if both fail.
    """
    if not os.path.exists(path):
        return None

    # Attempt 1: pydicom
    try:
        import pydicom

        dcm = pydicom.dcmread(path)
        img = dcm.pixel_array
        return img
    except (ImportError, Exception):
        pass

    # Attempt 2: OpenCV
    try:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            return img
    except Exception:
        pass

    return None


def _get_sorted_dicom_files(directory):
    """
    Returns a list of DICOM files in the directory, sorted numerically
    by the integer number in the filename (e.g., Image-10.dcm).
    """
    if not os.path.isdir(directory):
        return []

    files = glob.glob(os.path.join(directory, "*.dcm"))

    # Sort by the integer found in the filename (e.g. Image-123.dcm -> 123)
    def extract_number(filepath):
        match = re.search(r"Image-(\d+)\.dcm", os.path.basename(filepath))
        return int(match.group(1)) if match else 0

    sorted_files = sorted(files, key=extract_number)
    return sorted_files


def get_middle_slice(directory):
    """
    Returns the path to the middle slice of the sorted DICOM files.
    This simple geometric heuristic is often more robust than content-based selection
    for registered datasets (Cite Lesson 00036, Lesson 00006).
    """
    sorted_files = _get_sorted_dicom_files(directory)

    if not sorted_files:
        return None

    # Simple median index
    middle_idx = len(sorted_files) // 2
    return sorted_files[middle_idx]
