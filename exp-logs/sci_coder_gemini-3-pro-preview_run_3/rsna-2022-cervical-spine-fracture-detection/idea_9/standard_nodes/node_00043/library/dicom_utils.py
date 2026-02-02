import os
import glob
import numpy as np
import pydicom
from library.config import Config


def load_scan(path):
    """
    Loads all DICOM files from a directory.

    Args:
        path (str): Path to the directory containing DICOM files.

    Returns:
        list: A list of pydicom datasets.
    """
    # Search for all files in the directory
    search_pattern = os.path.join(path, "*")
    files = glob.glob(search_pattern)

    slices = []
    for f in files:
        try:
            # Read the DICOM file
            # force=True allows reading files missing the standard preamble
            dcm = pydicom.dcmread(f, force=True)

            # Verify pixel data exists
            if hasattr(dcm, "pixel_array"):
                slices.append(dcm)
        except Exception:
            # Skip files that cannot be read or are not valid DICOMs
            continue

    return slices


def sort_slices(slices):
    """
    Sorts a list of DICOM slices strictly by ImagePositionPatient z-coordinate.

    Args:
        slices (list): List of pydicom datasets.

    Returns:
        list: Sorted list of pydicom datasets.
    """
    # Filter slices that have the required ImagePositionPatient attribute
    valid_slices = [s for s in slices if hasattr(s, "ImagePositionPatient")]

    # Sort by the Z coordinate (index 2 of ImagePositionPatient)
    # We cast to float to ensure correct numerical sorting
    valid_slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

    return valid_slices


def pixels_to_hu(slices):
    """
    Converts a list of sorted DICOM slices into a 3D numpy array of Hounsfield Units.

    Args:
        slices (list): Sorted list of pydicom datasets.

    Returns:
        np.ndarray: 3D numpy array (Depth, Height, Width) in HU.
    """
    images = []
    for s in slices:
        # Convert to float32 to prevent overflow during slope/intercept application
        image = s.pixel_array.astype(np.float32)

        # Retrieve Rescale Slope and Intercept, defaulting to identity if missing
        slope = float(getattr(s, "RescaleSlope", 1))
        intercept = float(getattr(s, "RescaleIntercept", 0))

        # Apply transformation: HU = pixel * slope + intercept
        if slope != 1:
            image = slope * image

        image += intercept

        images.append(image)

    if not images:
        return np.array([])

    # Stack 2D slices into a 3D volume
    return np.stack(images)


def apply_window(image, center, width):
    """
    Applies a window to the HU image and normalizes to 0-255 uint8.

    Args:
        image (np.ndarray): Input image in HU.
        center (float): Window center (level).
        width (float): Window width.

    Returns:
        np.ndarray: Windowed image in uint8 [0, 255].
    """
    # Calculate window boundaries
    img_min = center - width // 2
    img_max = center + width // 2

    # Clip values to the window range
    windowed = np.clip(image, img_min, img_max)

    # Normalize to [0, 1]
    if img_max != img_min:
        windowed = (windowed - img_min) / (img_max - img_min)
    else:
        windowed = windowed - img_min

    # Scale to [0, 255] and convert to uint8
    windowed = windowed * 255.0

    return windowed.astype(np.uint8)
