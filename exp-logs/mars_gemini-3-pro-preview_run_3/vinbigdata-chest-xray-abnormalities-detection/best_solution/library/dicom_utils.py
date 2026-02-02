import os
import numpy as np
import pydicom
import cv2
from library.config import IMG_SIZE


def process_dicom_image(file_path, target_size=IMG_SIZE, fix_monochrome=True):
    """
    Reads a DICOM file, handles photometric interpretation, normalizes to 0-255,
    and resizes to the target size.

    Args:
        file_path (str): Path to the DICOM file.
        target_size (int): Target resolution (width and height).
        fix_monochrome (bool): Whether to fix MONOCHROME1 interpretation.

    Returns:
        np.ndarray: Processed image array (uint8) of shape (target_size, target_size).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"DICOM file not found: {file_path}")

    try:
        # Read DICOM file
        # stop_before_pixels=False ensures we read the pixel data
        ds = pydicom.dcmread(file_path, stop_before_pixels=False)

        # Extract pixel array
        pixel_array = ds.pixel_array.astype(np.float32)

        # Handle Photometric Interpretation
        # MONOCHROME1: 0 is White, Max is Black.
        # We generally want 0 to be Black (Air) and Max to be White (Bone/Dense tissue)
        if fix_monochrome and hasattr(ds, "PhotometricInterpretation"):
            if ds.PhotometricInterpretation == "MONOCHROME1":
                pixel_array = np.max(pixel_array) - pixel_array

        # Normalize to 0-255 range
        pixel_min = np.min(pixel_array)
        pixel_max = np.max(pixel_array)

        if pixel_max > pixel_min:
            pixel_array = (pixel_array - pixel_min) / (pixel_max - pixel_min)
            pixel_array = pixel_array * 255.0
        else:
            # Handle cases where the image is constant (e.g., all black)
            pixel_array = np.zeros_like(pixel_array)

        # Convert to uint8
        pixel_array = pixel_array.astype(np.uint8)

        # Resize image
        if target_size is not None:
            pixel_array = cv2.resize(
                pixel_array, (target_size, target_size), interpolation=cv2.INTER_LINEAR
            )

        return pixel_array

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        # Return a black image of target size in case of error to prevent pipeline crash
        if target_size is not None:
            return np.zeros((target_size, target_size), dtype=np.uint8)
        return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)


def save_image(image_array, output_path):
    """
    Saves a numpy array as an image file using OpenCV.

    Args:
        image_array (np.ndarray): Image array to save.
        output_path (str): Full path where the image should be saved.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save image
    cv2.imwrite(output_path, image_array)
