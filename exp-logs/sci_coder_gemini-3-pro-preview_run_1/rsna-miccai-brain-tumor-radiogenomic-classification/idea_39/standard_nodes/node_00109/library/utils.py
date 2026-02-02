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


def calculate_modality_com(directory, offset_ratio=0.0):
    """
    Calculates the Z-axis Center of Mass (CoM) of the brain tissue in the given directory
    and returns the path to the slice at 'offset_ratio' relative to the CoM.

    Args:
        directory (str): Path to the folder containing DICOM files for one modality.
        offset_ratio (float): Relative offset from CoM.
                              -0.1 means 10% of brain depth below CoM.
                              0.0 means at the CoM.
                              0.1 means 10% of brain depth above CoM.

    Returns:
        str: Path to the selected DICOM file.
    """
    sorted_files = _get_sorted_dicom_files(directory)

    if not sorted_files:
        return None

    # If only one file, return it
    if len(sorted_files) == 1:
        return sorted_files[0]

    # Scan volume to find brain tissue (pixels > 0)
    z_indices = []
    masses = []

    for i, fpath in enumerate(sorted_files):
        img = read_dicom_image(fpath)
        if img is None:
            continue

        # Calculate 'mass' as number of non-zero pixels (binary mask area)
        # We assume background is 0
        mass = np.count_nonzero(img)

        if mass > 0:
            z_indices.append(i)
            masses.append(mass)

    # Fallback: If no brain tissue found (empty images), use geometric center of file list
    if not z_indices:
        middle_idx = len(sorted_files) // 2
        return sorted_files[middle_idx]

    # Convert to numpy for vectorized math
    z_indices = np.array(z_indices)
    masses = np.array(masses)

    # Calculate Z-axis Center of Mass (weighted average of indices)
    total_mass = np.sum(masses)
    z_com = np.sum(z_indices * masses) / total_mass

    # Calculate Brain Depth (span of slices containing brain)
    # Using min and max index where brain was seen
    z_min = z_indices[0]
    z_max = z_indices[-1]
    depth = z_max - z_min

    # Determine target index
    # Target = CoM + (Offset * Depth)
    target_z = z_com + (offset_ratio * depth)

    # Round to nearest integer index
    target_idx = int(np.round(target_z))

    # Clip to valid range of available files
    target_idx = max(0, min(target_idx, len(sorted_files) - 1))

    return sorted_files[target_idx]
