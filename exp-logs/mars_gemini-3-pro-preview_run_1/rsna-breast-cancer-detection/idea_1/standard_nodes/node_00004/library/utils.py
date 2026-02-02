import os
import numpy as np
import cv2
import torch
import rasterio
from library.config import Config


def load_image(file_path):
    """
    Loads an image from the given relative path using rasterio.
    Handles DICOM and JPEG2000 formats commonly found in mammography.

    Args:
        file_path (str): Relative path to the image (e.g., 'train_images/123/456.dcm')

    Returns:
        np.ndarray: The image data as a numpy array. Returns a blank array on failure.
    """
    full_path = os.path.join(Config.INPUT_DIR, file_path)

    if not os.path.exists(full_path):
        return np.zeros(Config.IMG_SIZE, dtype=np.uint8)

    try:
        # rasterio is capable of reading DICOM/JP2 formats
        with rasterio.open(full_path) as src:
            # Mammograms are single channel, read the first band
            img = src.read(1)
            return img
    except Exception:
        # Return blank image if file is corrupt or unreadable
        return np.zeros(Config.IMG_SIZE, dtype=np.uint8)


def get_roi_crop(img):
    """
    Crops the Region of Interest (ROI) - the breast tissue - from the background.
    Uses Otsu's thresholding to identify the tissue mask.

    Args:
        img (np.ndarray): Input image.

    Returns:
        np.ndarray: Cropped image containing the breast tissue.
    """
    if img is None or img.size == 0:
        return np.zeros(Config.IMG_SIZE, dtype=np.uint8)

    # Normalize to 0-255 uint8 for OpenCV processing
    img_min = img.min()
    img_max = img.max()

    if img_max == img_min:
        return img  # Empty or flat image

    img_uint8 = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)

    # Binarize using Otsu's thresholding to separate tissue from background
    _, binary = cv2.threshold(img_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return img

    # Find the largest contour by area (assumed to be the breast)
    c = max(contours, key=cv2.contourArea)

    # Get bounding box
    x, y, w, h = cv2.boundingRect(c)

    # Crop the original image
    crop = img[y : y + h, x : x + w]
    return crop


def process_image(file_path):
    """
    Full processing pipeline: Load -> ROI Crop -> Resize -> Normalize.

    Args:
        file_path (str): Relative path to the image.

    Returns:
        np.ndarray: Processed image tensor of shape (H, W) with values in [0, 1].
    """
    # 1. Load
    img = load_image(file_path)

    # 2. Crop ROI
    img = get_roi_crop(img)

    # 3. Resize
    target_h, target_w = Config.IMG_SIZE
    img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    # 4. Normalize to [0, 1]
    img = img.astype(np.float32)
    img_min = img.min()
    img_max = img.max()

    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min)
    else:
        img = img - img_min  # Zero out

    return img


def probabilistic_f1(y_true, y_pred, epsilon=1e-7):
    """
    Computes the Probabilistic F1 Score (pF1).

    Formula:
        pF1 = 2 * (pPrecision * pRecall) / (pPrecision + pRecall)
        pPrecision = pTP / (pTP + pFP)
        pRecall = pTP / (TP + FN)

    Args:
        y_true: Ground truth labels (0 or 1). Tensor or Array.
        y_pred: Predicted probabilities (0 to 1). Tensor or Array.
        epsilon: Small constant to avoid division by zero.

    Returns:
        float: The pF1 score.
    """
    # Ensure inputs are torch tensors
    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true)
    if not isinstance(y_pred, torch.Tensor):
        y_pred = torch.tensor(y_pred)

    # Cast to float
    y_true = y_true.float()
    y_pred = y_pred.float()

    # Flatten tensors
    y_true = y_true.view(-1)
    y_pred = y_pred.view(-1)

    # Calculate Probabilistic True Positives (pTP)
    # pTP = Sum(y_true * y_pred)
    pTP = (y_true * y_pred).sum()

    # Calculate Probabilistic False Positives (pFP)
    # pFP = Sum((1 - y_true) * y_pred)
    pFP = ((1 - y_true) * y_pred).sum()

    # Calculate Total Positives (TP + FN)
    # This is simply the sum of ground truth positives
    total_positives = y_true.sum()

    # Calculate pPrecision
    # pPrecision = pTP / (pTP + pFP)
    # Note: pTP + pFP = Sum(y_pred)
    p_precision = pTP / (pTP + pFP + epsilon)

    # Calculate pRecall
    # pRecall = pTP / (TP + FN)
    p_recall = pTP / (total_positives + epsilon)

    # Calculate pF1
    f1 = 2 * (p_precision * p_recall) / (p_precision + p_recall + epsilon)

    return f1.item()
