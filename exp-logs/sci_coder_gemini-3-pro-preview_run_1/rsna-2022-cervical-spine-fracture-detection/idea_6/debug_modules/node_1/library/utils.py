import os
import numpy as np
import cv2
import pydicom
from library import config


def load_dicom_array(path, size=None):
    """
    Reads a DICOM file and returns the pixel array converted to Hounsfield Units.

    Args:
        path (str): Path to the .dcm file.
        size (tuple or int, optional): Target size (width, height) or single int for square.
                                       Defaults to None (original size).

    Returns:
        np.ndarray: 2D numpy array of the image.
    """
    try:
        ds = pydicom.dcmread(path)
        # Convert to float to avoid overflow/underflow during rescaling
        img = ds.pixel_array.astype(np.float32)

        # Apply Rescale Slope and Intercept to convert to Hounsfield Units (HU)
        slope = getattr(ds, "RescaleSlope", 1.0)
        intercept = getattr(ds, "RescaleIntercept", 0.0)

        if slope != 1.0:
            img = img * slope
        if intercept != 0.0:
            img = img + intercept

        # Resize if requested
        if size is not None:
            if isinstance(size, int):
                size = (size, size)
            # cv2.resize expects (width, height), shape is (height, width)
            # For square resize it doesn't matter, but good to be precise
            img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)

        return img

    except Exception as e:
        print(f"Error loading DICOM file {path}: {e}")
        # Return a zero array of requested size or default 512x512 as fallback
        fallback_size = (
            size
            if size is not None
            else (config.FULL_IMAGE_SIZE, config.FULL_IMAGE_SIZE)
        )
        if isinstance(fallback_size, int):
            fallback_size = (fallback_size, fallback_size)
        return np.zeros(fallback_size, dtype=np.float32)


def apply_windowing(image, center, width):
    """
    Applies windowing to the CT image to highlight specific structures (e.g., bone).
    Maps the range [center - width/2, center + width/2] to [0, 1].

    Args:
        image (np.ndarray): Input image (in HU).
        center (float): Window center level.
        width (float): Window width.

    Returns:
        np.ndarray: Windowed image normalized to [0, 1].
    """
    lower = center - width / 2
    upper = center + width / 2

    # Clip values to the window
    img_windowed = np.clip(image, lower, upper)

    # Normalize to [0, 1]
    # Avoid division by zero
    if upper - lower != 0:
        img_windowed = (img_windowed - lower) / (upper - lower)
    else:
        img_windowed = img_windowed - lower  # Should be 0

    return img_windowed


def crop_to_roi(image, center_yx, size):
    """
    Crops the image around a specific center point (y, x).
    Pads with zeros if the crop area extends outside the image boundaries.

    Args:
        image (np.ndarray): Input 2D image.
        center_yx (tuple): Center coordinates (y, x).
        size (int): Output height and width (square crop).

    Returns:
        np.ndarray: Cropped image of shape (size, size).
    """
    h, w = image.shape[:2]
    cy, cx = center_yx

    half = size // 2

    # Calculate crop coordinates (top-left and bottom-right)
    # Using round to nearest integer for center
    y1 = int(np.round(cy - half))
    y2 = y1 + size
    x1 = int(np.round(cx - half))
    x2 = x1 + size

    # Calculate padding amounts
    pad_top = max(0, -y1)
    pad_bottom = max(0, y2 - h)
    pad_left = max(0, -x1)
    pad_right = max(0, x2 - w)

    # Calculate valid slice coordinates within the original image
    slice_y1 = max(0, y1)
    slice_y2 = min(h, y2)
    slice_x1 = max(0, x1)
    slice_x2 = min(w, x2)

    # Extract the valid crop
    if slice_y1 >= slice_y2 or slice_x1 >= slice_x2:
        # Crop is completely outside image
        cropped = np.zeros((size, size), dtype=image.dtype)
    else:
        crop_part = image[slice_y1:slice_y2, slice_x1:slice_x2]

        # Pad to desired size
        if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
            cropped = cv2.copyMakeBorder(
                crop_part,
                pad_top,
                pad_bottom,
                pad_left,
                pad_right,
                cv2.BORDER_CONSTANT,
                value=0,
            )
        else:
            cropped = crop_part

    # Ensure output shape is exactly (size, size)
    if cropped.shape[0] != size or cropped.shape[1] != size:
        cropped = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_LINEAR)

    return cropped


def save_nifti_slice(image, save_path):
    """
    Saves a 2D image slice as a NIfTI file for debugging purposes.
    If nibabel is not available, saves as .npy.

    Args:
        image (np.ndarray): 2D image array.
        save_path (str): Output path (ending in .nii or .nii.gz).
    """
    try:
        import nibabel as nib

        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # NIfTI images are typically 3D+. Expand dims if 2D.
        if image.ndim == 2:
            # Add Z dimension
            image_nii = image[:, :, np.newaxis]
        else:
            image_nii = image

        # Create identity affine matrix
        affine = np.eye(4)

        # Create NIfTI image object
        nifti_img = nib.Nifti1Image(image_nii, affine)

        # Save
        nib.save(nifti_img, save_path)

    except ImportError:
        # Fallback if nibabel is missing
        npy_path = os.path.splitext(save_path)[0] + ".npy"
        np.save(npy_path, image)
    except Exception as e:
        print(f"Failed to save NIfTI slice to {save_path}: {e}")
