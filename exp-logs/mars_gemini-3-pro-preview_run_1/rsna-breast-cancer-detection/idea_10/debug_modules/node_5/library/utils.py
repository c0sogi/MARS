import os
import random
import numpy as np
import torch
import cv2
import rasterio
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
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def probabilistic_f1(y_true, y_pred, beta=1):
    """
    Calculates the Probabilistic F1 score (pF1) as defined in the task.

    pF1 = 2 * (pPrecision * pRecall) / (pPrecision + pRecall)

    Where:
        pPrecision = pTP / (pTP + pFP)
        pRecall = pTP / (TP + FN)
        pTP = Sum(y_pred * y_true)
        pFP = Sum(y_pred * (1 - y_true))
        TP + FN = Sum(y_true)

    Args:
        y_true: Ground truth labels (0 or 1).
        y_pred: Predicted probabilities [0, 1].
        beta: Beta value for F-score (default 1).

    Returns:
        pF1 score (float).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Epsilon to prevent division by zero
    epsilon = 1e-7

    # Probabilistic True Positives
    p_tp = np.sum(y_pred * y_true)

    # Probabilistic False Positives
    p_fp = np.sum(y_pred * (1 - y_true))

    # Total Actual Positives (Sum of binary ground truth)
    # As per task description: denominator of pRecall is TP + FN
    total_positives = np.sum(y_true)

    # Probabilistic Precision
    p_precision = p_tp / (p_tp + p_fp + epsilon)

    # Probabilistic Recall
    p_recall = p_tp / (total_positives + epsilon)

    # Calculate pF1
    beta_sq = beta**2
    numerator = (1 + beta_sq) * p_precision * p_recall
    denominator = (beta_sq * p_precision) + p_recall

    if denominator == 0:
        return 0.0

    pf1 = numerator / (denominator + epsilon)

    return pf1


def load_image(path):
    """
    Loads an image from the given path, handling DICOM/JP2 formats via cv2 or rasterio.
    Enforces 'Fail Loudly' policy for missing/corrupt files.
    Normalizes pixel values to [0, 1].

    Args:
        path: Relative or absolute path to the image file.

    Returns:
        numpy array: 2D image (H, W) of type float32, normalized to [0, 1].
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image file not found at: {path}")

    img = None

    # Attempt 1: OpenCV
    # cv2.IMREAD_UNCHANGED is crucial for preserving 16-bit depth in medical images
    try:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    except Exception:
        img = None

    # Attempt 2: Rasterio
    # Useful for formats cv2 might miss, particularly JPEG 2000 embedded in DICOM
    if img is None:
        try:
            with rasterio.open(path) as src:
                img = src.read(1)  # Read the first band
        except Exception:
            img = None

    if img is None:
        raise ValueError(
            f"Failed to load image at {path}. File may be corrupt or format unsupported."
        )

    # Ensure Image is 2D (H, W)
    if len(img.shape) == 3:
        if img.shape[2] == 3:
            # Convert RGB/BGR to Gray
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif img.shape[2] == 4:
            # Convert BGRA to Gray
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        elif img.shape[2] == 1:
            img = img[:, :, 0]

    # Convert to float32 for processing
    img = img.astype(np.float32)

    # Min-Max Normalization to [0, 1]
    # This scales the pixel intensities (density) to a range suitable for CNNs
    # while preserving relative contrast.
    min_val = img.min()
    max_val = img.max()

    if max_val - min_val > 1e-6:
        img = (img - min_val) / (max_val - min_val)
    else:
        # If image is constant (e.g., all black), return zeros
        img = np.zeros_like(img)

    return img
