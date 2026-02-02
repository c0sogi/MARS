import os
import random
import numpy as np
import torch
import pydicom
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_dicom(path):
    """
    Reads a DICOM file and returns the pixel array converted to Hounsfield Units (HU).

    Args:
        path (str): Path to the .dcm file.

    Returns:
        np.ndarray: 2D numpy array of the image in Hounsfield Units.
    """
    try:
        dcm = pydicom.dcmread(path)
        pixel_array = dcm.pixel_array.astype(np.float32)

        # Apply Rescale Slope and Intercept to convert to HU
        intercept = getattr(dcm, "RescaleIntercept", 0)
        slope = getattr(dcm, "RescaleSlope", 1)

        if slope != 1:
            pixel_array = slope * pixel_array

        pixel_array += intercept

        return pixel_array
    except Exception as e:
        # Fallback for corrupt or missing files, though metadata checks should prevent this
        print(f"Error loading DICOM {path}: {e}")
        return np.zeros((512, 512), dtype=np.float32)


def get_weighted_log_loss(y_pred, y_true):
    """
    Computes the weighted multi-label logarithmic loss for the competition.

    The loss is calculated per row (label) and then averaged.
    Weights are applied such that 'patient_overall' is weighted more highly (7.0)
    than specific vertebrae (1.0).

    Args:
        y_pred (torch.Tensor or np.ndarray): Predicted probabilities (0-1). Shape (N, 8).
            Expected column order: [C1, C2, C3, C4, C5, C6, C7, patient_overall]
        y_true (torch.Tensor or np.ndarray): Ground truth labels (0 or 1). Shape (N, 8).

    Returns:
        float: The calculated weighted log loss.
    """
    # Convert numpy arrays to torch tensors if necessary
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.from_numpy(y_pred)
    if isinstance(y_true, np.ndarray):
        y_true = torch.from_numpy(y_true)

    # Ensure data is on the correct device and type
    device = y_pred.device
    y_pred = y_pred.float()
    y_true = y_true.float().to(device)

    # Clip predictions to avoid log(0)
    epsilon = 1e-15
    y_pred = torch.clamp(y_pred, epsilon, 1 - epsilon)

    # Define weights: C1-C7 = 1.0, patient_overall = 7.0
    # This reflects the competition logic where the patient level outcome is weighted
    # equal to the sum of the vertebrae outcomes.
    weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 7.0], device=device)

    # Calculate Binary Cross Entropy
    # L = - [y * log(p) + (1-y) * log(1-p)]
    loss = -(y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))

    # Apply weights
    # Broadcasting weights (8,) over loss (N, 8)
    weighted_loss = loss * weights

    # Average across all rows (all N*8 elements)
    # The metric specifies "loss is averaged across all rows".
    return weighted_loss.mean().item()
