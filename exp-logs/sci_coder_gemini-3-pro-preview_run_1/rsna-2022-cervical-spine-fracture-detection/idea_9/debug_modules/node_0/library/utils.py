import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

# Attempt to import pydicom for DICOM handling
try:
    import pydicom
except ImportError:
    pydicom = None

from library.config import Config


class WeightedLogLoss(nn.Module):
    """
    Weighted multi-label logarithmic loss.
    Implements the competition metric where 'patient_overall' is weighted more highly
    than specific fracture sub-types.
    """

    def __init__(self, weights=None):
        super().__init__()
        # Default weights based on competition description:
        # 1.0 for C1-C7 (indices 0-6)
        # 7.0 for patient_overall (index 7)
        if weights is None:
            self.weights = torch.tensor([1.0] * 7 + [7.0])
        else:
            self.weights = torch.tensor(weights)

    def forward(self, logits, targets):
        """
        Calculates the weighted binary cross entropy loss.

        Args:
            logits (torch.Tensor): Predicted logits of shape (batch_size, 8).
            targets (torch.Tensor): Ground truth labels of shape (batch_size, 8).

        Returns:
            torch.Tensor: Scalar loss value averaged across the batch.
        """
        # Ensure weights are on the correct device
        device = logits.device
        weights = self.weights.to(device)

        # Compute BCE with logits (numerically stable)
        # reduction='none' allows us to apply specific weights per class/column
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        # Apply weights
        weighted_loss = loss * weights

        # Average across all entries.
        # Since the metric is defined as "averaged across all rows" (where a row is a single prediction),
        # and we have weighted the importance of specific rows (classes), the mean of the weighted
        # tensor approximates the weighted average log loss.
        return weighted_loss.mean()


def load_dicom(path):
    """
    Reads a DICOM file and returns the pixel array converted to Hounsfield Units (HU).

    Args:
        path (str): Path to the .dcm file.

    Returns:
        np.ndarray: 2D numpy array of the image in HU (float32).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"DICOM file not found: {path}")

    if pydicom is None:
        # Fallback mechanism if pydicom is not available
        # Attempts to read as a standard image (e.g., if files were converted but kept extension)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ImportError(
                "pydicom is not installed and cv2 failed to read the file."
            )
        return img.astype(np.float32)

    try:
        dcm = pydicom.dcmread(path)
        pixel_array = dcm.pixel_array.astype(np.float32)

        # Convert to Hounsfield Units (HU)
        # HU = pixel_value * slope + intercept
        intercept = getattr(dcm, "RescaleIntercept", 0)
        slope = getattr(dcm, "RescaleSlope", 1)

        image = pixel_array * slope + intercept
        return image
    except Exception as e:
        raise RuntimeError(f"Failed to read DICOM file {path}: {e}")


def apply_windowing(image, center=Config.WINDOW_CENTER, width=Config.WINDOW_WIDTH):
    """
    Applies windowing to the CT image to highlight specific structures (e.g., bone).

    Args:
        image (np.ndarray): Input image in HU.
        center (float): Window center level.
        width (float): Window width.

    Returns:
        np.ndarray: Windowed image normalized to range [0, 1].
    """
    lower = center - width / 2
    upper = center + width / 2

    # Clip values to the window range
    image = np.clip(image, lower, upper)

    # Normalize to [0, 1]
    image = (image - lower) / (upper - lower)

    return image.astype(np.float32)


def get_roi_coordinates(mask, padding=0):
    """
    Calculates the bounding box coordinates from a binary segmentation mask.

    Args:
        mask (np.ndarray): 2D binary mask (0 background, >0 foreground).
        padding (int): Padding to add around the bounding box.

    Returns:
        tuple: (ymin, ymax, xmin, xmax) coordinates.
    """
    # Check if mask is empty
    if np.sum(mask) == 0:
        h, w = mask.shape
        return 0, h, 0, w

    # Find rows and cols with non-zero values
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    # Get indices
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]

    h, w = mask.shape

    # Apply padding and clip to image boundaries
    ymin = max(0, int(ymin - padding))
    ymax = min(h, int(ymax + padding))
    xmin = max(0, int(xmin - padding))
    xmax = min(w, int(xmax + padding))

    return ymin, ymax, xmin, xmax


def save_features(features, path):
    """
    Saves a numpy array of features to the specified path, creating directories if needed.

    Args:
        features (np.ndarray): The feature array to save.
        path (str): The destination file path (e.g., .npy).
    """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    np.save(path, features)
