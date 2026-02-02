import os
import random
import numpy as np
import pandas as pd
import torch
import pydicom

import cv2
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


def read_dicom(path, window_center=400, window_width=1800, target_size=None):
    """
    Reads a DICOM file, applies windowing, and normalizes.

    Args:
        path (str): Path to the DICOM file.
        window_center (int): Window center for CT windowing.
        window_width (int): Window width for CT windowing.
        target_size (tuple, optional): (height, width) to resize the image.

    Returns:
        np.ndarray: Processed image (2D array), normalized to [0, 1].
    """
    try:
        dcm = pydicom.dcmread(path)
        img = dcm.pixel_array.astype(np.float32)

        # Apply Rescale Slope and Intercept if present to get Hounsfield Units
        if hasattr(dcm, "RescaleSlope") and hasattr(dcm, "RescaleIntercept"):
            slope = float(dcm.RescaleSlope)
            intercept = float(dcm.RescaleIntercept)
            img = img * slope + intercept

        # Apply Windowing
        img_min = window_center - window_width // 2
        img_max = window_center + window_width // 2
        img = np.clip(img, img_min, img_max)

        # Normalize to [0, 1]
        if img_max != img_min:
            img = (img - img_min) / (img_max - img_min)
        else:
            img = np.zeros_like(img)

        # Resize if requested
        if target_size is not None:
            if img.shape[0] != target_size[0] or img.shape[1] != target_size[1]:
                img = cv2.resize(
                    img,
                    (target_size[1], target_size[0]),
                    interpolation=cv2.INTER_LINEAR,
                )

        return img

    except Exception as e:
        # Return a black image of target size or default 512x512 in case of error
        size = target_size if target_size else (512, 512)
        return np.zeros(size, dtype=np.float32)


def weighted_log_loss(y_true, y_pred, weights=None):
    """
    Calculates the weighted multi-label logarithmic loss.

    Args:
        y_true (np.ndarray or torch.Tensor): True labels (N, 8).
        y_pred (np.ndarray or torch.Tensor): Predicted probabilities (N, 8).
        weights (list or np.ndarray, optional): Weights for each column.

    Returns:
        float: The weighted log loss.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    # Default weights: C1-C7 = 1.0, patient_overall = 7.0 (weighted more highly)
    if weights is None:
        weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 7.0])

    if len(weights) != y_true.shape[1]:
        weights = np.ones(y_true.shape[1])

    # L_ij = -w_j * [y_ij * log(p_ij) + (1 - y_ij) * log(1 - p_ij)]
    loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    weighted_loss = loss * weights

    return np.mean(weighted_loss)


def save_checkpoint(model, optimizer, epoch, path, scaler=None, metric=None):
    """Saves model checkpoint."""
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if scaler:
        state["scaler_state_dict"] = scaler.state_dict()
    if metric is not None:
        state["metric"] = metric

    torch.save(state, path)


def load_checkpoint(model, optimizer, path, scaler=None, device="cpu"):
    """Loads model checkpoint."""
    if not os.path.exists(path):
        return None

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scaler and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    return checkpoint.get("epoch", 0)


def save_to_cache(data, filename):
    """
    Saves data to the cache directory defined in Config.
    Uses .npy for numpy arrays, .parquet for DataFrames, .pth for others.
    """
    filepath = os.path.join(Config.CACHE_DIR, filename)

    if isinstance(data, pd.DataFrame):
        data.to_parquet(filepath, index=False)
    elif isinstance(data, np.ndarray):
        np.save(filepath, data)
    else:
        torch.save(data, filepath)


def load_from_cache(filename):
    """
    Loads data from the cache directory.
    """
    filepath = os.path.join(Config.CACHE_DIR, filename)
    if not os.path.exists(filepath):
        return None

    if filename.endswith(".parquet"):
        return pd.read_parquet(filepath)
    elif filename.endswith(".npy"):
        return np.load(filepath)
    else:
        return torch.load(filepath)


class AverageMeter:
    """Computes and stores the average and current value."""

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
