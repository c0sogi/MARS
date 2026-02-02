import os
import re
import glob
import random
import numpy as np
import torch
import pydicom
import cv2
from library.config import IMAGE_SIZE, SEED


def set_seed(seed=SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
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


def _extract_image_id(filename):
    """
    Extracts the integer ID from a DICOM filename (e.g., 'Image-123.dcm' -> 123).
    Returns None if the pattern does not match.
    """
    match = re.search(r"Image-(\d+)\.dcm", filename)
    if match:
        return int(match.group(1))
    return None


def load_dicom_slice(path, target_size=IMAGE_SIZE):
    """
    Reads a DICOM file and returns the pixel array.

    Args:
        path (str): Full path to the DICOM file.
        target_size (tuple, optional): (H, W) to resize the image to.
                                       If None, returns original size.
                                       Defaults to IMAGE_SIZE from config.

    Returns:
        np.ndarray or None: The pixel array (resized if target_size is provided),
                            or None if file missing/unreadable.
    """
    if not os.path.exists(path):
        return None

    try:
        ds = pydicom.dcmread(path)
        img = ds.pixel_array

        # Resize if dimensions differ and target_size is specified
        if target_size is not None and (
            img.shape[0] != target_size[0] or img.shape[1] != target_size[1]
        ):
            img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)

        return img
    except Exception:
        return None


def get_brain_depth_range(modality_path):
    """
    Iterates through DICOM files in the modality directory to find the z-range
    containing actual brain tissue (non-zero pixels). This defines the
    Brain-Centric ROI.

    Args:
        modality_path (str): Path to the specific modality directory (e.g., .../FLAIR).

    Returns:
        tuple: (min_idx, max_idx) representing the start and end indices (inclusive)
               of the brain tissue. Returns (0, 0) if no valid data found.
    """
    # List all DICOM files
    files = glob.glob(os.path.join(modality_path, "*.dcm"))
    if not files:
        return 0, 0

    # Map indices to full paths
    idx_map = {}
    for f in files:
        idx = _extract_image_id(os.path.basename(f))
        if idx is not None:
            idx_map[idx] = f

    sorted_indices = sorted(idx_map.keys())
    if not sorted_indices:
        return 0, 0

    # Find first non-empty slice (scan from start)
    min_idx = sorted_indices[0]
    for idx in sorted_indices:
        # Load without resizing for speed check
        img = load_dicom_slice(idx_map[idx], target_size=None)
        if img is not None and np.max(img) > 0:
            min_idx = idx
            break

    # Find last non-empty slice (scan from end)
    max_idx = sorted_indices[-1]
    for idx in reversed(sorted_indices):
        img = load_dicom_slice(idx_map[idx], target_size=None)
        if img is not None and np.max(img) > 0:
            max_idx = idx
            break

    return min_idx, max_idx


def read_dicom_slab(modality_path, center_idx, slab_depth=3):
    """
    Reads a slab of consecutive DICOM slices centered at center_idx.
    Used to create the [z-1, z, z+1] input volume.

    Args:
        modality_path (str): Path to the modality directory.
        center_idx (int): The index of the center slice.
        slab_depth (int): Number of slices in the slab (should be odd).

    Returns:
        np.ndarray: 3D array of shape (H, W, slab_depth).
                    Missing slices are padded with zeros.
    """
    half_depth = slab_depth // 2
    # Generate indices: e.g., for depth 3, center 10 -> [9, 10, 11]
    indices = range(center_idx - half_depth, center_idx + half_depth + 1)

    slices = []
    for idx in indices:
        # Construct expected filename
        filename = f"Image-{idx}.dcm"
        file_path = os.path.join(modality_path, filename)

        img = load_dicom_slice(file_path, target_size=IMAGE_SIZE)

        if img is None:
            # Pad with zeros if slice is missing (e.g., out of bounds)
            img = np.zeros(IMAGE_SIZE, dtype=np.float32)

        slices.append(img)

    # Stack along the last dimension -> (H, W, D)
    slab = np.stack(slices, axis=-1)
    return slab


def independent_slab_normalize(slab):
    """
    Applies Min-Max normalization to the slab independently.
    This preserves the relative intensity differences between slices in the slab
    while scaling the data to [0, 1] for the neural network.

    Args:
        slab (np.ndarray): Input slab array.

    Returns:
        np.ndarray: Normalized slab in range [0, 1] with dtype float32.
    """
    slab = slab.astype(np.float32)
    min_val = np.min(slab)
    max_val = np.max(slab)

    if max_val > min_val:
        slab = (slab - min_val) / (max_val - min_val)
    else:
        # If flat (e.g. all zeros), return zeros
        slab = np.zeros_like(slab)

    return slab
