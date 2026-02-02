import os
import re
import numpy as np
import pydicom
import cv2
from library import config


def get_sorted_file_paths(file_paths):
    """
    Sorts a list of file paths based on the integer index found in the filename.
    Example: 'Image-10.dcm' is sorted as 10, not lexicographically.

    Args:
        file_paths (list): List of relative file path strings.

    Returns:
        list: Sorted list of file paths.
    """

    def extract_number(path):
        # Extract the filename from the path
        filename = os.path.basename(path)
        # Find all digit sequences
        matches = re.findall(r"\d+", filename)
        if matches:
            # Typically the last number is the instance number in BraTS naming convention
            # or the only number. We take the last one found to be safe,
            # assuming format like Image-123.dcm
            return int(matches[-1])
        return 0

    return sorted(file_paths, key=extract_number)


def load_dicom_slice(rel_path):
    """
    Loads a single DICOM slice, converts to float, and resizes to configuration size.

    Args:
        rel_path (str): Relative path to the DICOM file from input root.

    Returns:
        np.ndarray: 2D numpy array of shape (IMG_SIZE, IMG_SIZE).
    """
    full_path = os.path.join(config.INPUT_DIR, rel_path)

    try:
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"File not found: {full_path}")

        dcm = pydicom.dcmread(full_path)
        img = dcm.pixel_array.astype(np.float32)

        # Resize to native resolution defined in config
        # cv2.resize expects (width, height)
        img_resized = cv2.resize(
            img, (config.IMG_SIZE, config.IMG_SIZE), interpolation=cv2.INTER_AREA
        )

        return img_resized

    except Exception as e:
        # In case of corruption or read error, return a black slice
        # This ensures the stack shape remains consistent
        # In a production pipeline, we might log this error
        return np.zeros((config.IMG_SIZE, config.IMG_SIZE), dtype=np.float32)


def load_and_process_modality_block(file_paths):
    """
    Loads a list of DICOM paths (representing a slice subset for one modality),
    stacks them, and performs View-Adaptive Per-Modality Normalization.

    Normalization is calculated using Min/Max statistics derived ONLY from
    the loaded block, ensuring independence from other views or modalities.

    Args:
        file_paths (list): List of relative file paths for the specific slices.

    Returns:
        np.ndarray: 3D numpy array of shape (D, H, W) where D is len(file_paths).
                    Values are normalized to [0, 1].
    """
    if not file_paths:
        # Return empty block if no paths provided (though caller should handle this)
        return np.zeros((0, config.IMG_SIZE, config.IMG_SIZE), dtype=np.float32)

    # Load all slices in the block
    slices = [load_dicom_slice(p) for p in file_paths]

    # Stack into (Depth, Height, Width)
    volume = np.stack(slices, axis=0)

    # View-Adaptive Per-Modality Normalization
    # Calculate stats only on this specific block
    min_val = np.min(volume)
    max_val = np.max(volume)

    # Avoid division by zero if the image is completely flat (e.g. all black)
    if max_val - min_val > 0:
        volume = (volume - min_val) / (max_val - min_val)
    else:
        # If flat, subtract min (likely 0) to get 0s
        volume = volume - min_val

    return volume
