import os
import re
import random
import numpy as np
import torch
import cv2
from library.config import SEED

# Attempt to import pydicom safely
try:
    import pydicom

    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False


def seed_everything(seed=SEED):
    """
    Sets the seed for generating random numbers to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_dicom_as_array(path):
    """
    Reads a DICOM file and returns the pixel array.
    Prioritizes pydicom, falls back to OpenCV.
    Returns None if reading fails.
    """
    if not os.path.exists(path):
        return None

    # Attempt 1: pydicom
    if HAS_PYDICOM:
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


def normalize_minmax(img):
    """
    Normalizes a numpy array to the range [0, 1] using float32.
    Handles constant images (max == min) by returning a zero array.
    """
    if img is None:
        return None

    img = img.astype(np.float32)
    min_val = np.min(img)
    max_val = np.max(img)

    if max_val > min_val:
        return (img - min_val) / (max_val - min_val)
    else:
        return np.zeros_like(img)


def get_sorted_file_list(directory):
    """
    Returns a sorted list of DICOM filenames from a directory.
    Sorts numerically based on the integer index in the filename
    (e.g., 'Image-10.dcm' comes after 'Image-2.dcm').
    """
    if not os.path.exists(directory):
        return []

    files = [f for f in os.listdir(directory) if f.endswith(".dcm")]

    def extract_number(filename):
        s = re.search(r"(\d+)", filename)
        return int(s.group(1)) if s else 0

    files.sort(key=extract_number)
    return files
