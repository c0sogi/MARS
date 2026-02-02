import os
import random
import numpy as np
import torch
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut

from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility using the Config method.
    """
    Config.seed_everything(seed)


def read_dicom(file_path, fix_monochrome=True, voi_lut=True):
    """
    Reads a DICOM file, applies VOI LUT, handles photometric interpretation,
    and normalizes the image to 8-bit [0, 255].

    Args:
        file_path (str): Path to the DICOM file.
        fix_monochrome (bool): Whether to fix MONOCHROME1 images.
        voi_lut (bool): Whether to apply VOI LUT.

    Returns:
        np.ndarray: The processed image array (H, W) in uint8.
    """
    try:
        dicom = pydicom.dcmread(file_path)

        # Apply VOI LUT if requested and available
        if voi_lut:
            data = apply_voi_lut(dicom.pixel_array, dicom)
        else:
            data = dicom.pixel_array

        # Handle MONOCHROME1 (0 is white) -> convert to MONOCHROME2 (0 is black)
        # Chest X-rays typically display air (low density) as black.
        if (
            fix_monochrome
            and getattr(dicom, "PhotometricInterpretation", "") == "MONOCHROME1"
        ):
            data = np.amax(data) - data

        # Normalize to [0, 255]
        data = data.astype(np.float32)
        if np.max(data) > np.min(data):
            data = data - np.min(data)
            data = data / np.max(data)
        else:
            # Handle constant image case
            data = data - np.min(data)

        data = (data * 255).astype(np.uint8)
        return data

    except Exception as e:
        # Fallback for corrupt files or read errors
        print(f"Warning: Failed to read DICOM {file_path}: {e}")
        # Return a blank image matching the target size to prevent pipeline crash
        return np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE), dtype=np.uint8)


def collate_fn(batch):
    """
    Custom collate function for DETR-style training.
    Handles variable number of bounding boxes by keeping targets as a list.

    Args:
        batch (list): List of tuples from the Dataset.
                      Expected formats: (image, target) or (image, target, image_id).

    Returns:
        tuple: (images, targets) or (images, targets, image_ids)
    """
    # Filter out None values if any (e.g. from failed reads)
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None

    # Check the structure of the first element to determine if IDs are present
    elem = batch[0]
    has_ids = len(elem) == 3

    if has_ids:
        images, targets, image_ids = zip(*batch)
    else:
        images, targets = zip(*batch)

    # Stack images into a single tensor (B, C, H, W)
    # Assumes images are already resized and converted to tensors in Dataset
    images = torch.stack(images, dim=0)

    # Targets remain a list of dictionaries/tensors as required by DETR
    targets = list(targets)

    if has_ids:
        return images, targets, list(image_ids)
    else:
        return images, targets
