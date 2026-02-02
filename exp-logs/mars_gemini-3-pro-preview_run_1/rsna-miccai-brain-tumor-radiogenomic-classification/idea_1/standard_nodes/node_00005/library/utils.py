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


def load_best_slice(directory_path):
    """
    Loads the 'best' slice from a directory based on variance (Cite solution_lesson_node_00002).

    1. Lists all .dcm files.
    2. Sorts them.
    3. Iterates through the middle 40% of slices.
    4. Selects the slice with the highest pixel variance.

    Args:
        directory_path (str): Path to the directory.

    Returns:
        np.ndarray: The pixel array of the selected slice.
    """
    fallback_shape = (Config.IMG_SIZE, Config.IMG_SIZE)

    if not os.path.exists(directory_path):
        return np.zeros(fallback_shape, dtype=np.uint8)

    files = [f for f in os.listdir(directory_path) if f.endswith(".dcm")]

    if not files:
        return np.zeros(fallback_shape, dtype=np.uint8)

    # Sort files numerically
    files.sort(key=_extract_number)

    # Focus on the middle 40% to avoid skull/neck (indices 30% to 70%)
    n_files = len(files)
    start_idx = int(n_files * 0.30)
    end_idx = int(n_files * 0.70)

    # Ensure valid range
    if start_idx >= end_idx:
        start_idx = 0
        end_idx = n_files

    candidate_files = files[start_idx:end_idx]

    best_img = None
    max_variance = -1.0

    for fname in candidate_files:
        full_path = os.path.join(directory_path, fname)
        img = read_dicom_file(full_path)

        if img is None:
            continue

        # Use variance as a proxy for information content/tumor texture
        # Cite solution_lesson_node_00002
        current_variance = np.var(img)

        if current_variance > max_variance:
            max_variance = current_variance
            best_img = img

    if best_img is None:
        return np.zeros(fallback_shape, dtype=np.uint8)

    return best_img
