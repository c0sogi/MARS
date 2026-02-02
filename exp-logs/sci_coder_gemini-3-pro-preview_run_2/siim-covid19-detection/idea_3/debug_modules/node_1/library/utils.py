import os
import cv2
import numpy as np
import pydicom
import torch
import random
from pydicom.pixel_data_handlers.util import apply_voi_lut
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def apply_clahe(img):
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to the image.
    Expects a uint8 numpy array.
    """
    if img is None:
        return None

    # Ensure image is uint8
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)

    # Create CLAHE object with standard parameters
    # clipLimit=2.0 and tileGridSize=(8,8) are common defaults for X-rays
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img)
    return img_clahe


def read_dicom(path, fix_monochrome=True):
    """
    Reads a DICOM file and converts it to a standard uint8 numpy array.
    Handles VOI LUT and Photometric Interpretation (MONOCHROME1/2).
    """
    try:
        dicom = pydicom.dcmread(path)

        # Apply VOI LUT if available (handles windowing)
        # This converts pixel_array to the display values
        if "VOILUTSequence" in dicom or "WindowCenter" in dicom:
            data = apply_voi_lut(dicom.pixel_array, dicom)
        else:
            data = dicom.pixel_array

        # Handle Photometric Interpretation
        # MONOCHROME1: 0 is white, 1 is black (need to invert for standard view)
        # MONOCHROME2: 0 is black, 1 is white (standard)
        if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
            data = np.amax(data) - data

        # Normalize to 0-255 range
        data = data.astype(np.float32)
        data = data - np.min(data)
        max_val = np.max(data)
        if max_val > 0:
            data = data / max_val

        data = (data * 255).astype(np.uint8)

        return data

    except Exception as e:
        # In case of corruption or read error
        print(f"Error reading DICOM file {path}: {e}")
        return None


def collate_fn(batch):
    """
    Custom collate function for the DataLoader.
    Handles variable number of bounding boxes in targets.

    Args:
        batch: List of tuples (image, target, image_id)
               image: Tensor of shape (C, H, W)
               target: Dict containing 'boxes', 'labels', etc.
               image_id: String

    Returns:
        images: Tensor of shape (B, C, H, W)
        targets: List of Dicts
        image_ids: List of Strings
    """
    # Filter out any None items if read_dicom failed (though Dataset usually handles this)
    batch = [b for b in batch if b[0] is not None]

    if len(batch) == 0:
        return torch.Tensor(), [], []

    images = []
    targets = []
    image_ids = []

    for b in batch:
        images.append(b[0])
        targets.append(b[1])
        image_ids.append(b[2])

    # Stack images into a single tensor
    # Assumes images are already resized to Config.IMG_SIZE in the Dataset
    images = torch.stack(images, dim=0)

    return images, targets, image_ids
