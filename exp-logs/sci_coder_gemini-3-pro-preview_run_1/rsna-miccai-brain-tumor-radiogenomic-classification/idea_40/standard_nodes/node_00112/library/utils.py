import os
import sys
import logging
import random
import numpy as np
import pandas as pd
import torch
import cv2
from sklearn.metrics import roc_auc_score
from library.config import (
    SEED,
    DEVICE,
    WORKING_DIR,
    CACHE_DIR,
    seed_everything as config_seed_everything,
)

# ==========================================
# General Utilities
# ==========================================


def seed_everything(seed=SEED):
    """
    Wrapper for the config seed_everything function.
    """
    config_seed_everything(seed)


def get_device():
    """
    Returns the configured torch device.
    """
    return DEVICE


def get_logger(name="train", log_file=None):
    """
    Configures and returns a logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Clear existing handlers to prevent duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Stream Handler (Console)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    # File Handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


class MetricMonitor:
    """
    A helper class to track and average metrics over batches.
    """

    def __init__(self, float_precision=4):
        self.reset()
        self.float_precision = float_precision

    def reset(self):
        self.metrics = {}

    def update(self, metric_name, val, n=1):
        val = float(val)  # Ensure value is a float
        if metric_name not in self.metrics:
            self.metrics[metric_name] = {"sum": 0.0, "count": 0}
        self.metrics[metric_name]["sum"] += val * n
        self.metrics[metric_name]["count"] += n

    def get_avg(self, metric_name):
        if metric_name not in self.metrics:
            return 0.0
        return self.metrics[metric_name]["sum"] / self.metrics[metric_name]["count"]

    def __str__(self):
        return " | ".join(
            [
                "{}: {:.{prec}f}".format(
                    name, self.get_avg(name), prec=self.float_precision
                )
                for name in self.metrics
            ]
        )


# ==========================================
# Image Processing Utilities
# ==========================================


def read_dicom_image(path):
    """
    Reads a DICOM file from the specified path.
    Tries pydicom first (if available), then falls back to OpenCV.
    Returns a numpy array or None if failure.
    """
    # Attempt 1: pydicom
    try:
        import pydicom

        dcm = pydicom.dcmread(path)
        img = dcm.pixel_array
        return img
    except (ImportError, Exception):
        pass

    # Attempt 2: OpenCV
    try:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            return img
    except Exception:
        pass

    return None


def min_max_scale(img):
    """
    Scales a numpy array to [0, 1].
    Handles cases where max == min (returns zero array).
    """
    img = img.astype(np.float32)
    min_val = img.min()
    max_val = img.max()

    if max_val - min_val > 0:
        return (img - min_val) / (max_val - min_val)
    else:
        return np.zeros_like(img)


def get_brain_bbox(volume_3d, threshold=0):
    """
    Calculates the 3D bounding box of the brain tissue (pixels > threshold).
    volume_3d: (Depth, Height, Width) numpy array.
    Returns: (z_min, z_max, y_min, y_max, x_min, x_max)
    """
    rows = np.any(volume_3d > threshold, axis=(1, 2))
    cols = np.any(volume_3d > threshold, axis=(0, 2))
    slices = np.any(volume_3d > threshold, axis=(0, 1))

    if not np.any(rows) or not np.any(cols) or not np.any(slices):
        # Empty volume or threshold too high
        d, h, w = volume_3d.shape
        return 0, d, 0, h, 0, w

    z_min, z_max = np.where(rows)[0][[0, -1]]
    y_min, y_max = np.where(cols)[0][[0, -1]]
    x_min, x_max = np.where(slices)[0][[0, -1]]

    return z_min, z_max, y_min, y_max, x_min, x_max


def get_center_of_mass_z(volume_3d, threshold=0):
    """
    Calculates the Center of Mass along the Z-axis (depth).
    Useful for finding the 'middle' of the brain biologically.
    """
    # Create a mask of the brain
    mask = (volume_3d > threshold).astype(np.float32)
    total_mass = np.sum(mask)

    if total_mass == 0:
        return volume_3d.shape[0] // 2

    # Z indices
    z_indices = np.arange(volume_3d.shape[0])
    # Sum mass per slice
    mass_per_slice = np.sum(mask, axis=(1, 2))

    com_z = np.sum(z_indices * mass_per_slice) / total_mass
    return int(round(com_z))


# ==========================================
# Caching Utilities
# ==========================================


def save_numpy_cache(data, filename, directory=CACHE_DIR):
    """
    Saves a numpy array to the cache directory.
    """
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, filename)
    np.save(path, data)


def load_numpy_cache(filename, directory=CACHE_DIR):
    """
    Loads a numpy array from the cache directory.
    Returns None if file does not exist.
    """
    path = os.path.join(directory, filename)
    if os.path.exists(path):
        return np.load(
            path, allow_pickle=True
        )  # allow_pickle needed for object arrays if any
    return None


def save_parquet_cache(df, filename, directory=CACHE_DIR):
    """
    Saves a pandas DataFrame to parquet.
    """
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, filename)
    df.to_parquet(path, index=False)


def load_parquet_cache(filename, directory=CACHE_DIR):
    """
    Loads a pandas DataFrame from parquet.
    """
    path = os.path.join(directory, filename)
    if os.path.exists(path):
        return pd.read_parquet(path)
    return None


# ==========================================
# Metrics & Submission
# ==========================================


def calculate_roc_auc(y_true, y_pred):
    """
    Calculates Area Under the ROC Curve.
    Handles edge cases where only one class is present in y_true.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Check for single class case
    if len(np.unique(y_true)) < 2:
        return 0.5

    return roc_auc_score(y_true, y_pred)


def save_predictions(ids, preds, path="./submission/submission.csv"):
    """
    Saves predictions to a CSV file in the required format.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    df = pd.DataFrame({"BraTS21ID": ids, "MGMT_value": preds})

    # Ensure BraTS21ID is formatted correctly (though int is usually fine)
    # The sample submission uses standard integers.

    df.to_csv(path, index=False)
    print(f"Predictions saved to {path}")
