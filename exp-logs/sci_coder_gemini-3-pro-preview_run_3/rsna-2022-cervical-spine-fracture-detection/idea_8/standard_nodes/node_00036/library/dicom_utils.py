import os
import glob
import numpy as np
import cv2
import warnings
from library.config import Config

# Attempt to import pydicom.
# It is strictly required for the physics-based preprocessing (HU conversion, Z-sorting).
try:
    import pydicom

    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False


def get_z_position(dcm):
    """
    Extracts Z-position from a pydicom dataset to ensure anatomical sorting.
    Prioritizes ImagePositionPatient[2], then SliceLocation, then InstanceNumber.
    """
    if hasattr(dcm, "ImagePositionPatient"):
        # ImagePositionPatient is usually a list/tuple of 3 floats
        return float(dcm.ImagePositionPatient[2])
    if hasattr(dcm, "SliceLocation"):
        return float(dcm.SliceLocation)
    if hasattr(dcm, "InstanceNumber"):
        return float(dcm.InstanceNumber)
    return 0.0


def convert_to_hu(dcm, pixel_array):
    """
    Converts raw pixel array to Hounsfield Units (HU) using RescaleSlope and RescaleIntercept.
    This normalizes pixel values across different scanners.
    """
    intercept = getattr(dcm, "RescaleIntercept", 0.0)
    slope = getattr(dcm, "RescaleSlope", 1.0)

    # Handle cases where these might be stored as lists/strings in some DICOM headers
    if isinstance(slope, (list, tuple)):
        slope = slope[0]
    if isinstance(intercept, (list, tuple)):
        intercept = intercept[0]

    return pixel_array * float(slope) + float(intercept)


def apply_bone_window(hu_image):
    """
    Applies standard bone window (Level=400, Width=1800) and normalizes to [0, 1].
    This highlights bone structures while suppressing soft tissue and air.
    """
    level = Config.WINDOW_LEVEL
    width = Config.WINDOW_WIDTH

    lower = level - width / 2
    upper = level + width / 2

    # Clip values to the window range
    img = np.clip(hu_image, lower, upper)

    # Normalize to 0-1 range
    if width > 0:
        img = (img - lower) / width
    else:
        img = np.zeros_like(img)

    return img


def load_scan(study_dir, resize_to=Config.IMAGE_SIZE):
    """
    Loads a DICOM scan from a directory, processing it into a 3D volume.

    Pipeline:
    1. Identify all DICOM files in the directory.
    2. Read metadata to extract Z-position and pixel data.
    3. Convert raw pixels to Hounsfield Units (HU).
    4. Apply Bone Windowing.
    5. Resize slice to target resolution.
    6. Sort slices anatomically by Z-position.
    7. Stack into a 3D numpy array.

    Args:
        study_dir (str): Path to the study directory containing .dcm files.
        resize_to (tuple): Target resolution (Height, Width). Defaults to Config.IMAGE_SIZE.

    Returns:
        np.ndarray: 3D volume of shape (Depth, Height, Width) with float32 values in [0, 1].
                    Returns an empty array if loading fails or directory is empty.
    """
    if not HAS_PYDICOM:
        warnings.warn(
            "pydicom module not found. Cannot load DICOM files. Returning empty volume."
        )
        return np.zeros((0, *resize_to), dtype=np.float32)

    if not os.path.exists(study_dir):
        return np.zeros((0, *resize_to), dtype=np.float32)

    # Find all DICOM files
    dicom_files = glob.glob(os.path.join(study_dir, "*.dcm"))

    # Fallback: some datasets might not have .dcm extension
    if not dicom_files:
        all_files = glob.glob(os.path.join(study_dir, "*"))
        dicom_files = [f for f in all_files if os.path.isfile(f)]

    if not dicom_files:
        return np.zeros((0, *resize_to), dtype=np.float32)

    slices = []

    for file_path in dicom_files:
        try:
            # Read DICOM file
            # stop_before_pixels=False ensures we read the pixel data
            dcm = pydicom.dcmread(file_path, stop_before_pixels=False)

            # Extract Z position for sorting
            z = get_z_position(dcm)

            # Get pixels
            try:
                pixel_array = dcm.pixel_array.astype(np.float32)
            except Exception:
                # This can happen if compression is not supported (e.g. missing pylibjpeg/gdcm)
                # We skip this slice rather than crashing the whole scan load
                continue

            # Convert to Hounsfield Units
            hu_img = convert_to_hu(dcm, pixel_array)

            # Apply Windowing
            windowed_img = apply_bone_window(hu_img)

            # Resize
            if resize_to:
                # cv2.resize expects (width, height)
                windowed_img = cv2.resize(
                    windowed_img,
                    (resize_to[1], resize_to[0]),
                    interpolation=cv2.INTER_LINEAR,
                )

            slices.append((z, windowed_img))

        except Exception:
            # Skip corrupt files or read errors
            continue

    if not slices:
        return np.zeros((0, *resize_to), dtype=np.float32)

    # Sort slices by Z position to ensure anatomical continuity
    slices.sort(key=lambda x: x[0])

    # Stack into 3D volume
    volume = np.stack([s[1] for s in slices])

    return volume.astype(np.float32)
