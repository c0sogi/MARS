import os
import cv2
import numpy as np
import torch
import random
from library.config import Config

# Attempt to import pydicom for robust DICOM reading
try:
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_voi_lut

    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False


def seed_everything(seed=42):
    """
    Seeds all random number generators for reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_dicom(path, fix_monochrome=True):
    """
    Reads a DICOM file and returns a normalized numpy array (H, W, 3).

    Args:
        path (str): Path to the DICOM file.
        fix_monochrome (bool): Whether to invert MONOCHROME1 images.

    Returns:
        np.ndarray: Image array with shape (H, W, 3) and dtype uint8 (0-255).
    """
    img = None

    # Attempt 1: Use pydicom (Preferred for medical accuracy)
    if HAS_PYDICOM:
        try:
            dcm = pydicom.dcmread(path)

            # Apply VOI LUT (Windowing) if available
            if "VOILUTSequence" in dcm or "WindowCenter" in dcm:
                img = apply_voi_lut(dcm.pixel_array, dcm)
            else:
                img = dcm.pixel_array

            # Fix Photometric Interpretation (MONOCHROME1 is inverted)
            if fix_monochrome and dcm.PhotometricInterpretation == "MONOCHROME1":
                img = np.amax(img) - img
        except Exception:
            # Fallback if pydicom fails on specific file
            img = None

    # Attempt 2: Use OpenCV (Fallback)
    if img is None:
        try:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        except Exception:
            pass

    # Safety check: if reading failed completely, return a black image
    if img is None:
        # Return a blank 1024x1024 image to prevent pipeline crash
        return np.zeros((1024, 1024, 3), dtype=np.uint8)

    # Normalize to 0-255 range
    if img.dtype != np.uint8:
        img = img.astype(np.float32)
        img_min = np.min(img)
        img_max = np.max(img)
        img = img - img_min
        if img_max > img_min:
            img = img / (img_max - img_min)
        img = (img * 255).astype(np.uint8)

    # Ensure 3 channels (RGB) for model compatibility
    if len(img.shape) == 2:
        img = np.stack([img, img, img], axis=-1)
    elif len(img.shape) == 3 and img.shape[2] == 1:
        img = np.concatenate([img, img, img], axis=-1)

    return img


def collate_fn(batch):
    """
    Custom collate function for object detection DataLoaders.
    Handles variable numbers of bounding boxes per image.

    Args:
        batch: List of tuples (image, target, image_id)

    Returns:
        Tuple of (images, targets, image_ids)
    """
    return tuple(zip(*batch))


def format_prediction_string(labels, boxes, scores):
    """
    Formats prediction results into the competition submission string format.

    Args:
        labels (list): List of class labels (strings) or IDs.
        boxes (list): List of bounding boxes [xmin, ymin, xmax, ymax].
        scores (list): List of confidence scores.

    Returns:
        str: Space-separated string "label score xmin ymin xmax ymax ..."
    """
    pred_strings = []
    for label, score, box in zip(labels, scores, boxes):
        xmin, ymin, xmax, ymax = box
        # Format: label score xmin ymin xmax ymax
        pred_strings.append(f"{label} {score:.6f} {xmin} {ymin} {xmax} {ymax}")

    return " ".join(pred_strings)
