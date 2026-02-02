import os
import random
import numpy as np
import torch
import cv2
import pandas as pd
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the seed for random number generators to ensure reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_dicom(path, fix_monochrome=True):
    """
    Reads a DICOM file and returns a numpy array (uint8).
    Handles MONOCHROME1 inversion and normalization.
    Falls back to cv2 if pydicom fails.
    """
    try:
        dicom = pydicom.dcmread(path)

        # Apply VOI LUT if available (handles windowing)
        if "VOILUTSequence" in dicom or "WindowCenter" in dicom:
            data = apply_voi_lut(dicom.pixel_array, dicom)
        else:
            data = dicom.pixel_array

        # Handle MONOCHROME1 (where 0 is white, we want 0 to be black)
        if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
            data = np.amax(data) - data

        # Normalize to 0-255
        data = data - np.min(data)
        data = data / np.max(data)
        data = (data * 255).astype(np.uint8)

        return data

    except Exception:
        # Fallback to OpenCV
        try:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Could not load image at {path}")
            return img
        except Exception as e:
            # Return a blank image or re-raise depending on strictness
            # Here we raise to avoid silent failures in training
            raise ValueError(f"Failed to read DICOM at {path}: {e}")


def collate_fn(batch):
    """
    Custom collate function for object detection.
    Batch is a list of tuples (image, target, image_id).
    Returns a tuple of lists.
    """
    return tuple(zip(*batch))


class AverageMeter(object):
    """
    Computes and stores the average and current value.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_cached_data(
    cache_name, generate_fn, load_cached_data=True, base_dir=Config.WORKING_DIR
):
    """
    Generic caching mechanism.

    Args:
        cache_name (str): Name of the cache file (e.g., 'processed_data').
        generate_fn (callable): Function that returns the data (pd.DataFrame or dict/list) if cache misses.
        load_cached_data (bool): Whether to attempt loading from cache.
        base_dir (str): Directory to store cache files.

    Returns:
        The data.
    """
    os.makedirs(base_dir, exist_ok=True)

    # Determine file extension based on expected return type logic or generic convention
    # We will support parquet for DataFrames and npy for generic numpy/dict structures wrapped in object
    parquet_path = os.path.join(base_dir, f"{cache_name}.parquet")
    npy_path = os.path.join(base_dir, f"{cache_name}.npy")

    # 1. Try to load
    if load_cached_data:
        if os.path.exists(parquet_path):
            return pd.read_parquet(parquet_path)
        elif os.path.exists(npy_path):
            return np.load(npy_path, allow_pickle=True).item()

    # 2. Generate
    data = generate_fn()

    # 3. Save
    if isinstance(data, pd.DataFrame):
        data.to_parquet(parquet_path)
    else:
        # Save as numpy object
        np.save(npy_path, data)

    return data


def format_prediction_string(labels, boxes, scores):
    """
    Formats predictions into the competition string format.

    Args:
        labels (list): List of class names or IDs.
        boxes (list): List of [xmin, ymin, xmax, ymax].
        scores (list): List of confidence scores.

    Returns:
        str: "class conf xmin ymin xmax ymax ..."
    """
    pred_strings = []
    for label, score, box in zip(labels, scores, boxes):
        xmin, ymin, xmax, ymax = box
        # Ensure box coordinates are valid
        pred_strings.append(f"{label} {score} {xmin} {ymin} {xmax} {ymax}")

    return " ".join(pred_strings)
