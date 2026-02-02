import os
import random
import numpy as np
import torch
import cv2
import pydicom
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def uniform_sample_indices(total_slices, num_samples=Config.NUM_SLICES):
    """
    Selects 'num_samples' indices uniformly distributed across the 10%-90% depth range.

    Args:
        total_slices (int): Total number of slices in the volume.
        num_samples (int): Number of slices to sample.

    Returns:
        np.ndarray: Array of integer indices.
    """
    if total_slices == 0:
        return np.array([], dtype=int)

    # Define the 10% to 90% range to avoid edge artifacts
    start_idx = int(total_slices * 0.1)
    end_idx = int(total_slices * 0.9)

    # Handle cases where the range is too small (e.g., very few slices)
    if end_idx <= start_idx:
        start_idx = 0
        end_idx = total_slices

    # Generate linearly spaced indices
    # We use end_idx - 1 because we want to include the upper bound in the sampling logic
    # but ensure it is a valid index (< total_slices).
    # If we used end_idx, and end_idx == total_slices, we'd get an out-of-bounds index.
    indices = np.linspace(start_idx, end_idx - 1, num_samples)

    # Round to nearest integer and cast
    indices = np.round(indices).astype(int)

    # Clip to ensure indices are strictly within valid bounds
    indices = np.clip(indices, 0, total_slices - 1)

    return indices


def global_normalize(volume):
    """
    Normalizes the 3D volume based on the global minimum and maximum pixel intensities.
    Scales values to [0, 1].

    Args:
        volume (np.ndarray): 3D array of pixel data.

    Returns:
        np.ndarray: Normalized volume.
    """
    if volume.size == 0:
        return volume

    v_min = np.min(volume)
    v_max = np.max(volume)

    if v_max - v_min > 0:
        volume = (volume - v_min) / (v_max - v_min)
    else:
        # If the volume is constant (e.g., all zeros), return zeros
        volume = np.zeros_like(volume)

    return volume


def load_dicom_volume(file_paths, image_size=Config.IMAGE_SIZE):
    """
    Loads a list of DICOM files, resizes them, and stacks them into a 3D volume.

    Args:
        file_paths (list): List of relative file paths to DICOM images.
        image_size (int): Target spatial dimension (H=W).

    Returns:
        np.ndarray: 3D array of shape (Depth, image_size, image_size).
    """
    volume = []

    # Iterate through the provided paths
    for rel_path in file_paths:
        full_path = os.path.join(Config.INPUT_DIR, rel_path)

        if not os.path.exists(full_path):
            continue

        try:
            # Read DICOM file
            dcm = pydicom.dcmread(full_path)
            img = dcm.pixel_array.astype(float)

            # Resize if necessary
            if img.shape[0] != image_size or img.shape[1] != image_size:
                img = cv2.resize(
                    img, (image_size, image_size), interpolation=cv2.INTER_AREA
                )

            volume.append(img)
        except Exception:
            # Skip corrupted or unreadable files
            continue

    # Handle case where no images were loaded
    if len(volume) == 0:
        return np.zeros((0, image_size, image_size), dtype=np.float32)

    return np.array(volume, dtype=np.float32)
