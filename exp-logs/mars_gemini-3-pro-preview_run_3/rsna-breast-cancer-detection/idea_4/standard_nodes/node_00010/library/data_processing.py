import os
import numpy as np
import cv2
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from library.config import Config


def load_dicom(path):
    """
    Reads a DICOM file and converts it to a standard 8-bit numpy array.

    Handles:
    - Windowing/Leveling via VOI LUT.
    - Photometric Interpretation (inverting MONOCHROME1).
    - Normalization to 0-255 range.

    Args:
        path (str): Path to the .dcm file.

    Returns:
        np.ndarray: 2D numpy array (uint8) containing the image data.
                    Returns None if loading fails.
    """
    if not os.path.exists(path):
        return None

    try:
        dicom = pydicom.dcmread(path)

        # Apply VOI LUT (Value of Interest Look-Up Table) if available.
        # This transforms the raw pixel data into the display values intended by the device.
        # It handles window center/width logic.
        if "VOILUTSequence" in dicom or "WindowCenter" in dicom:
            data = apply_voi_lut(dicom.pixel_array, dicom)
        else:
            data = dicom.pixel_array

        # Handle Photometric Interpretation
        # MONOCHROME1: 0 is White (dense), 1 is Black.
        # MONOCHROME2: 0 is Black, 1 is White (dense).
        # We want dense tissue to be bright (standard for CNNs).
        if dicom.PhotometricInterpretation == "MONOCHROME1":
            data = np.amax(data) - data

        # Normalize to 0-255 uint8
        data = data.astype(np.float32)

        # Avoid division by zero if image is completely flat
        max_val = np.max(data)
        min_val = np.min(data)

        if max_val > min_val:
            data = (data - min_val) / (max_val - min_val)
        else:
            data = np.zeros_like(data)

        data = (data * 255).astype(np.uint8)
        return data

    except Exception as e:
        # In a production/competition setting, we might log this.
        # For now, return None to let the dataset handler manage the missing file.
        return None


def crop_breast_roi(image):
    """
    Crops the image to the bounding box of the breast tissue.

    Uses Otsu's thresholding to find the breast area and removes
    the large background regions.

    Args:
        image (np.ndarray): Input 2D image (uint8).

    Returns:
        np.ndarray: Cropped image. Returns original if cropping fails.
    """
    try:
        # Binarize the image to separate tissue from background
        # Otsu's thresholding is generally robust for this
        _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find contours
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return image

        # Assume the largest contour is the breast
        c = max(contours, key=cv2.contourArea)

        # Get bounding box
        x, y, w, h = cv2.boundingRect(c)

        # Crop
        cropped = image[y : y + h, x : x + w]

        # Safety check: if crop is too small (e.g. noise), return original
        if cropped.size == 0 or w < 20 or h < 20:
            return image

        return cropped

    except Exception:
        return image


def preprocess_image(
    image, target_height=Config.IMG_HEIGHT, target_width=Config.IMG_WIDTH
):
    """
    Prepares an image for the model:
    1. Resizes to target dimensions.
    2. Converts to 3 channels (RGB) by replication.
    3. Normalizes to [0, 1] float32.

    Args:
        image (np.ndarray): Input image (uint8, 2D).
        target_height (int): Height for resizing.
        target_width (int): Width for resizing.

    Returns:
        np.ndarray: Processed image array of shape (H, W, 3), float32, range [0, 1].
    """
    # Resize
    # cv2.resize expects (width, height)
    image = cv2.resize(
        image, (target_width, target_height), interpolation=cv2.INTER_LINEAR
    )

    # Convert to 3 channels (EfficientNet expects RGB)
    # We replicate the grayscale channel
    image = np.stack([image, image, image], axis=-1)

    # Normalize to [0, 1]
    image = image.astype(np.float32) / 255.0

    return image
