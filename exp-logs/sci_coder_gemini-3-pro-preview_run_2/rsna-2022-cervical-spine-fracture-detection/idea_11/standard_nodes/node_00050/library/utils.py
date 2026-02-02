import os
import random
import numpy as np
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


def load_dicom(path, size=None):
    """
    Loads a DICOM file, converts to Hounsfield Units (HU), applies bone windowing,
    and normalizes to [0, 1].

    Args:
        path (str): Path to the DICOM file.
        size (tuple, optional): Target size (height, width).

    Returns:
        np.ndarray: Processed image as a float32 numpy array.
    """
    try:
        dicom = pydicom.dcmread(path)

        # Get pixel array
        img = dicom.pixel_array.astype(np.float32)

        # Convert to Hounsfield Units
        intercept = getattr(dicom, "RescaleIntercept", 0)
        slope = getattr(dicom, "RescaleSlope", 1)
        img = img * slope + intercept

        # Apply Bone Windowing
        # Center = 400, Width = 1800 (Range: -500 to 1300)
        # This highlights bone structures while suppressing soft tissue
        window_center = 400
        window_width = 1800

        img_min = window_center - window_width // 2
        img_max = window_center + window_width // 2

        img = np.clip(img, img_min, img_max)

        # Normalize to [0, 1]
        img = (img - img_min) / (img_max - img_min)

        # Resize if requested
        if size is not None:
            # cv2.resize expects (width, height)
            img = cv2.resize(img, (size[1], size[0]))

        return img

    except Exception as e:
        # Fallback for missing or corrupt files: return a black image
        if size is not None:
            return np.zeros(size, dtype=np.float32)
        return np.zeros((512, 512), dtype=np.float32)


def weighted_log_loss(y_true, y_pred):
    """
    Calculates the weighted multi-label logarithmic loss.

    Weights:
        - patient_overall: 7.0
        - C1-C7: 1.0 each

    This weighting scheme ensures the 'patient_overall' label contributes
    roughly 50% to the total loss, prioritizing accurate patient-level diagnosis.

    Args:
        y_true (np.ndarray): Binary targets, shape (N, 8).
                             Columns: C1, C2, C3, C4, C5, C6, C7, patient_overall.
        y_pred (np.ndarray): Probabilities, shape (N, 8).

    Returns:
        float: The weighted logarithmic loss.
    """
    # Clip predictions to prevent log(0)
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    # Define weights
    weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 7.0])

    # Calculate binary cross entropy for each class
    # shape: (N, 8)
    bce = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    # Average loss per class (across samples)
    # shape: (8,)
    mean_bce_per_class = np.mean(bce, axis=0)

    # Compute weighted average across classes
    final_loss = np.sum(mean_bce_per_class * weights) / np.sum(weights)

    return final_loss
