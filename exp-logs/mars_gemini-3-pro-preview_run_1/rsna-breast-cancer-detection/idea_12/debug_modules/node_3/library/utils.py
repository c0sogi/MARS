import os
import random
import numpy as np
import torch
import cv2
import rasterio


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def probabilistic_f1(y_true, y_pred, epsilon=1e-7):
    """
    Computes the Probabilistic F1 score (pF1).

    Formula:
        pF1 = 2 * (pPrecision * pRecall) / (pPrecision + pRecall)
        pPrecision = pTP / (pTP + pFP)
        pRecall = pTP / (TP + FN)

    Args:
        y_true (array-like): Ground truth labels (binary 0/1).
        y_pred (array-like): Predicted probabilities (0.0 to 1.0).
        epsilon (float): Small constant to prevent division by zero.

    Returns:
        float: The probabilistic F1 score.
    """
    # Convert inputs to numpy arrays if they are tensors
    if hasattr(y_true, "cpu"):
        y_true = y_true.detach().cpu().numpy()
    if hasattr(y_pred, "cpu"):
        y_pred = y_pred.detach().cpu().numpy()

    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)

    # Probabilistic True Positives (pTP)
    # Sum of predicted probability for actual positive cases
    p_tp = np.sum(y_true * y_pred)

    # Probabilistic False Positives (pFP)
    # Sum of predicted probability for actual negative cases
    p_fp = np.sum((1.0 - y_true) * y_pred)

    # Total Positives (TP + FN)
    # The actual count of positive cases in the ground truth
    total_positives = np.sum(y_true)

    # Calculate pPrecision
    # pTP / (pTP + pFP)
    p_precision = p_tp / (p_tp + p_fp + epsilon)

    # Calculate pRecall
    # pTP / (TP + FN)
    p_recall = p_tp / (total_positives + epsilon)

    # Calculate pF1
    p_f1 = 2 * (p_precision * p_recall) / (p_precision + p_recall + epsilon)

    return p_f1


def robust_image_loader(path):
    """
    Attempts to load an image from the given path using cv2 or rasterio.
    Raises an exception if the file cannot be loaded.

    Args:
        path (str): Path to the image file.

    Returns:
        np.ndarray: The loaded image.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image file not found at: {path}")

    # Attempt 1: OpenCV
    # cv2 can often load standard formats and sometimes DICOM/JP2 depending on build
    try:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            # If loaded as BGR (3 channels), convert to RGB
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
    except Exception:
        # Continue to next method if cv2 fails
        pass

    # Attempt 2: Rasterio
    # Rasterio (via GDAL) is robust for various formats including JPEG 2000
    try:
        with rasterio.open(path) as src:
            # rasterio reads as (Count, Height, Width)
            img = src.read()

            # Move channels to the last dimension: (Height, Width, Count)
            img = np.moveaxis(img, 0, -1)

            # If single channel, squeeze to (Height, Width)
            if img.shape[-1] == 1:
                img = np.squeeze(img, axis=-1)

            return img
    except Exception:
        pass

    # If all attempts fail, raise an exception (Fail Loudly)
    raise IOError(f"Failed to load image at {path}. Tried cv2 and rasterio.")
