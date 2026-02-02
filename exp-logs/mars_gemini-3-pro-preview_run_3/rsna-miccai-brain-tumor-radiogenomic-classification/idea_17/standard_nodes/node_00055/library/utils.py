import os
import numpy as np
import pydicom
import cv2
from library.config import Config


def generate_strided_indices(num_files):
    """
    Generates indices for View A (Even) and View B (Odd) based on the
    High-Density Uniform Sampling strategy (32 slices from 10-90% depth).

    Args:
        num_files (int): Total number of files in the modality folder.

    Returns:
        dict: Dictionary containing 'view_a' and 'view_b' lists of indices.
    """
    if num_files == 0:
        return {"view_a": [], "view_b": []}

    # Define depth range (10% to 90%) to avoid start/end artifacts
    start = int(num_files * 0.1)
    end = int(num_files * 0.9)

    # Handle cases with very few slices where range might be invalid
    if end <= start:
        start = 0
        end = num_files - 1

    # Generate 32 uniformly distributed indices across the valid range
    # Config.TOTAL_SLICES is set to 32 in config.py
    indices = np.linspace(start, end, Config.TOTAL_SLICES)
    indices = np.round(indices).astype(int)

    # Clip to ensure valid bounds (safety check)
    indices = np.clip(indices, 0, num_files - 1)

    # Deterministic Strided View Split
    # View A: Even indices (0, 2, 4...) -> 16 slices
    # View B: Odd indices (1, 3, 5...) -> 16 slices
    view_a = indices[0::2].tolist()
    view_b = indices[1::2].tolist()

    return {"view_a": view_a, "view_b": view_b}


def load_dicom_volume(paths, img_size=Config.IMG_SIZE):
    """
    Loads a list of DICOM files, stacks them into a volume, and performs
    Global Volumetric Normalization.

    Args:
        paths (list): List of relative file paths to DICOM files.
        img_size (int): Spatial dimension to resize images to.

    Returns:
        np.ndarray: Normalized volume of shape (Depth, img_size, img_size) in float32.
    """
    slices = []

    for path in paths:
        full_path = os.path.join(Config.INPUT_DIR, path)
        try:
            # Read DICOM file
            dcm = pydicom.dcmread(full_path)
            img = dcm.pixel_array.astype(float)

            # Handle Photometric Interpretation:
            # Ensure 0 represents background/air. MONOCHROME1 has 0 as white, so we invert.
            if (
                hasattr(dcm, "PhotometricInterpretation")
                and dcm.PhotometricInterpretation == "MONOCHROME1"
            ):
                img = np.max(img) - img

            # Resize slice to target dimensions
            if img.shape[0] != img_size or img.shape[1] != img_size:
                img = cv2.resize(
                    img, (img_size, img_size), interpolation=cv2.INTER_AREA
                )

            slices.append(img)

        except Exception as e:
            # Robustness: append a black slice if file is corrupt or missing
            # This prevents the pipeline from crashing on a single bad file
            slices.append(np.zeros((img_size, img_size), dtype=float))

    if not slices:
        return np.zeros((0, img_size, img_size), dtype=np.float32)

    # Stack slices to form (Depth, Height, Width) volume
    volume = np.stack(slices, axis=0)

    # Global Volumetric Normalization
    # Normalize pixel intensities based on the global minimum and maximum of the loaded volume.
    # This preserves the relative contrast between slices (e.g., empty slices remain dark).
    v_min = np.min(volume)
    v_max = np.max(volume)

    if v_max - v_min > 0:
        volume = (volume - v_min) / (v_max - v_min)
    else:
        # Avoid division by zero for constant volumes (e.g., all black)
        volume = np.zeros_like(volume)

    return volume.astype(np.float32)
