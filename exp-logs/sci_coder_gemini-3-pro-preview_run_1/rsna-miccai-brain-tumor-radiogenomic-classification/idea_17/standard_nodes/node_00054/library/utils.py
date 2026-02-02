import os
import re
import random
import numpy as np
import torch
import cv2
import pydicom


def seed_everything(seed):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_dicom_slice(path):
    """
    Reads a DICOM file from the given path and returns it as a float32 numpy array.
    Attempts to use pydicom first, then falls back to OpenCV.

    Args:
        path (str): Full path to the .dcm file.

    Returns:
        np.ndarray: Image data as float32 array.
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

    if img is None:
        # If reading fails, raise an error or return a zero array.
        # Given strict data requirements, we raise an error to avoid silent failures.
        raise ValueError(f"Could not read DICOM file: {path}")

    # Convert to float32 for high-fidelity processing
    return img.astype(np.float32)


def get_brain_roi_depth(folder_path):
    """
    Iterates through a sorted list of DICOM files in a modality folder to find the
    anatomical depth (start and end indices) of the brain tissue.

    The function performs a natural sort on filenames (e.g., Image-1, Image-2, ... Image-10)
    and scans from the beginning and end of the stack to find the first non-zero images.

    Args:
        folder_path (str): Path to the directory containing .dcm files for a specific modality.

    Returns:
        tuple: (start_index, end_index, sorted_filenames)
    """
    if not os.path.exists(folder_path):
        return 0, 0, []

    # List all DICOM files
    files = [f for f in os.listdir(folder_path) if f.endswith(".dcm")]

    if not files:
        return 0, 0, []

    # Natural sort key function
    def extract_number(f):
        # Extracts '10' from 'Image-10.dcm'
        match = re.search(r"(\d+)", f)
        return int(match.group(1)) if match else 0

    sorted_files = sorted(files, key=extract_number)

    # Initialize indices
    start_idx = 0
    end_idx = len(sorted_files) - 1

    # Find start index (first slice with signal)
    # We scan forward until we find a non-empty image
    for i, fname in enumerate(sorted_files):
        path = os.path.join(folder_path, fname)
        try:
            img = load_dicom_slice(path)
            if np.max(img) > 0:
                start_idx = i
                break
        except Exception:
            continue

    # Find end index (last slice with signal)
    # We scan backward until we find a non-empty image
    for i in range(len(sorted_files) - 1, -1, -1):
        fname = sorted_files[i]
        path = os.path.join(folder_path, fname)
        try:
            img = load_dicom_slice(path)
            if np.max(img) > 0:
                end_idx = i
                break
        except Exception:
            continue

    # Handle edge case where no signal is found or start > end
    if start_idx > end_idx:
        start_idx = 0
        end_idx = len(sorted_files) - 1

    return start_idx, end_idx, sorted_files
