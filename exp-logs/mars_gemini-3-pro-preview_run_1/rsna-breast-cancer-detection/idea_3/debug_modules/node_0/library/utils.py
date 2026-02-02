import os
import random
import numpy as np
import torch
import cv2
import rasterio
from library.config import SEED


def seed_everything(seed=SEED):
    """
    Seeds all random number generators to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_dicom(path):
    """
    Loads a mammogram image from a DICOM file.
    Prioritizes rasterio to handle JPEG 2000 compression often found in these DICOMs.
    Falls back to OpenCV if rasterio fails.

    Args:
        path (str): Path to the .dcm file.

    Returns:
        np.ndarray or None: The image data as a numpy array, or None if loading fails.
    """
    if not os.path.exists(path):
        return None

    img = None

    # Attempt 1: Rasterio (Handles JPEG 2000 / embedded formats well)
    try:
        with rasterio.open(path) as src:
            img = src.read(1)
    except Exception:
        pass

    # Attempt 2: OpenCV (Fallback)
    if img is None:
        try:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        except Exception:
            pass

    return img


def crop_roi(image, threshold=0):
    """
    Crops the Region of Interest (ROI) by removing the background.
    Assumes the background has low pixel values (typically 0).

    Args:
        image (np.ndarray): The input image.
        threshold (int): Pixel value threshold to distinguish background.
                         Pixels > threshold are considered foreground.

    Returns:
        np.ndarray: The cropped image containing the ROI.
    """
    if image is None:
        return None

    if image.size == 0:
        return image

    # Create binary mask
    mask = image > threshold

    # If the image is entirely background, return original
    if not np.any(mask):
        return image

    # Find bounding box of the foreground
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Crop to bounding box
    cropped = image[rmin : rmax + 1, cmin : cmax + 1]

    return cropped


def pf1_score(labels, preds):
    """
    Computes the Probabilistic F1 Score (pF1).

    pF1 = 2 * (pPrecision * pRecall) / (pPrecision + pRecall)

    Where:
        pPrecision = pTP / (pTP + pFP)
        pRecall = pTP / (TP + FN)
        pTP = Sum(preds * labels)
        pFP = Sum(preds * (1 - labels))

    Args:
        labels (np.ndarray or torch.Tensor): Ground truth binary labels (0 or 1).
        preds (np.ndarray or torch.Tensor): Predicted probabilities (0 to 1).

    Returns:
        float: The probabilistic F1 score.
    """
    # Convert tensors to numpy if necessary
    if hasattr(labels, "detach"):
        labels = labels.detach().cpu().numpy()
    if hasattr(preds, "detach"):
        preds = preds.detach().cpu().numpy()

    # Flatten arrays to ensure 1D
    y_true = labels.flatten()
    y_pred = preds.flatten()

    # Calculate Probabilistic True Positives (pTP)
    pTP = np.sum(y_pred * y_true)

    # Calculate Probabilistic False Positives (pFP)
    pFP = np.sum(y_pred * (1 - y_true))

    # Calculate Total Positives (Actual Positives count)
    # TP + FN is simply the sum of true labels
    total_positives = np.sum(y_true)

    # Calculate pPrecision
    # Denominator is pTP + pFP = Sum(preds)
    predicted_mass = pTP + pFP

    if predicted_mass == 0:
        pPrecision = 0.0
    else:
        pPrecision = pTP / predicted_mass

    # Calculate pRecall
    if total_positives == 0:
        pRecall = 0.0
    else:
        pRecall = pTP / total_positives

    # Calculate pF1
    if pPrecision + pRecall == 0:
        pF1 = 0.0
    else:
        pF1 = 2 * (pPrecision * pRecall) / (pPrecision + pRecall)

    return pF1
