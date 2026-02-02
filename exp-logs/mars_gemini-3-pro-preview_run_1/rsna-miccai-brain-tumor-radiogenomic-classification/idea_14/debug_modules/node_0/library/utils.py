import os
import glob
import re
import random
import numpy as np
import cv2
import torch
from library.config import Config


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _natural_sort_key(s):
    """
    Sorts strings containing numbers naturally (e.g., Image-2.dcm before Image-10.dcm).
    """
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split("([0-9]+)", s)
    ]


def read_dicom_file(path):
    """
    Reads a DICOM file and returns a numpy array.
    Tries pydicom first (standard for DICOM), then falls back to cv2.
    """
    # Attempt 1: pydicom
    try:
        import pydicom

        dcm = pydicom.dcmread(path)
        img = dcm.pixel_array
        return img
    except (ImportError, Exception):
        pass

    # Attempt 2: cv2 (IMREAD_UNCHANGED to preserve depth)
    try:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            return img
    except Exception:
        pass

    return None


def normalize_min_max(img):
    """
    Applies Min-Max scaling to [0, 1] and converts to float32.
    """
    img = img.astype(np.float32)
    min_val = np.min(img)
    max_val = np.max(img)

    # Avoid division by zero
    if max_val - min_val > 0:
        img = (img - min_val) / (max_val - min_val)
    else:
        img = np.zeros_like(img)

    return img


def load_dicom_slab(modality_path, delta_offset, depth=3):
    """
    Loads a slab of DICOM images from a directory using Independent Heuristic Alignment.

    1. Finds all DICOM files in the directory.
    2. Sorts them naturally.
    3. Finds the median index.
    4. Selects a center index = median + delta_offset.
    5. Extracts 'depth' slices centered at that index.
    6. Resizes and normalizes each slice.

    Args:
        modality_path (str): Path to the directory containing DICOM files.
        delta_offset (int): Offset from the median index (0=center, negative=lower, positive=upper).
        depth (int): Number of consecutive slices to extract.

    Returns:
        np.ndarray: A numpy array of shape (depth, H, W) with float32 values in [0, 1].
    """
    # Initialize empty slab (Depth, H, W)
    slab = np.zeros((depth, Config.IMAGE_SIZE, Config.IMAGE_SIZE), dtype=np.float32)

    if not os.path.exists(modality_path):
        return slab

    # List files
    files = glob.glob(os.path.join(modality_path, "*.dcm"))
    if not files:
        return slab

    # Sort naturally (Image-1, Image-2, ..., Image-10)
    files = sorted(files, key=lambda x: _natural_sort_key(os.path.basename(x)))

    num_files = len(files)
    median_idx = num_files // 2

    # Calculate target center index
    center_idx = median_idx + delta_offset

    # Calculate start index (inclusive)
    # For depth 3 centered at C: [C-1, C, C+1], so start is C - 1
    start_idx = center_idx - (depth // 2)

    for i in range(depth):
        file_idx = start_idx + i

        # Check bounds
        if 0 <= file_idx < num_files:
            file_path = files[file_idx]
            img = read_dicom_file(file_path)

            if img is not None:
                # Resize to target size
                if img.shape != (Config.IMAGE_SIZE, Config.IMAGE_SIZE):
                    try:
                        img = cv2.resize(
                            img,
                            (Config.IMAGE_SIZE, Config.IMAGE_SIZE),
                            interpolation=cv2.INTER_AREA,
                        )
                    except Exception:
                        # Fallback if resize fails, keep zero slice
                        continue

                # Normalize to [0, 1]
                img = normalize_min_max(img)

                # Assign to slab
                slab[i] = img
            else:
                # If read fails, leave as zeros
                pass
        else:
            # If out of bounds (padding), leave as zeros
            pass

    return slab
