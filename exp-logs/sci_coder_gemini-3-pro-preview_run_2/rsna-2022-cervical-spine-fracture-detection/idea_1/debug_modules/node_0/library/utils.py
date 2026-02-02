import os
import re
import numpy as np
import pydicom
import cv2
from library.config import Config


def natural_sort_key(s):
    """
    Generates a key for natural sorting of strings containing numbers.
    Splits the string into text and numeric parts to ensure '10.dcm'
    comes after '2.dcm' instead of '1.dcm'.
    """
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split("([0-9]+)", s)
    ]


def get_study_paths(study_dir):
    """
    Retrieves and sorts all DICOM file paths from a given study directory.

    Args:
        study_dir (str): The path to the directory containing DICOM files.

    Returns:
        list: A sorted list of full file paths to the .dcm files.
    """
    if not os.path.exists(study_dir):
        return []

    files = [f for f in os.listdir(study_dir) if f.endswith(".dcm")]
    files.sort(key=natural_sort_key)

    full_paths = [os.path.join(study_dir, f) for f in files]
    return full_paths


def load_dicom_and_process(
    path,
    size=Config.IMG_SIZE,
    window_center=Config.BONE_WINDOW_CENTER,
    window_width=Config.BONE_WINDOW_WIDTH,
):
    """
    Loads a DICOM file, converts it to Hounsfield Units, applies windowing,
    normalizes, and resizes the image.

    Args:
        path (str): Path to the DICOM file.
        size (int): The target height and width for resizing.
        window_center (int): The center of the windowing range (HU).
        window_width (int): The width of the windowing range (HU).

    Returns:
        np.ndarray: The processed 2D image array with shape (size, size)
                    and values in range [0, 1]. Returns a zero array on failure.
    """
    try:
        dicom = pydicom.dcmread(path)

        # Access pixel data
        img = dicom.pixel_array.astype(np.float32)

        # Convert to Hounsfield Units (HU)
        # HU = pixel_value * slope + intercept
        slope = getattr(dicom, "RescaleSlope", 1)
        intercept = getattr(dicom, "RescaleIntercept", 0)
        img = img * slope + intercept

        # Apply Windowing
        # Clip values to the window range
        img_min = window_center - window_width // 2
        img_max = window_center + window_width // 2
        img = np.clip(img, img_min, img_max)

        # Normalize to [0, 1]
        if img_max != img_min:
            img = (img - img_min) / (img_max - img_min)
        else:
            img = np.zeros_like(img)

        # Resize image
        if size is not None:
            img = cv2.resize(img, (size, size))

        return img

    except Exception as e:
        # Return a blank image in case of corruption or read errors
        # to ensure the pipeline continues running.
        if size is not None:
            return np.zeros((size, size), dtype=np.float32)
        else:
            return np.zeros((512, 512), dtype=np.float32)
