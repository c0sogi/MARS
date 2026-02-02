import os
import numpy as np
import pydicom
import torch
import torch.nn.functional as F
from library.config import Config


def apply_windowing(image, center, width):
    """
    Applies windowing to the image to focus on specific structures (e.g., bone).

    Args:
        image (np.ndarray): Input image array (in Hounsfield Units).
        center (float): Window center.
        width (float): Window width.

    Returns:
        np.ndarray: Windowed image normalized to [0, 1].
    """
    lower = center - width / 2
    upper = center + width / 2

    # Clip pixel values to the window range
    image = np.clip(image, lower, upper)

    # Normalize to [0, 1]
    # Avoid division by zero if width is 0 (unlikely for valid windowing)
    if width > 0:
        image = (image - lower) / (upper - lower)
    else:
        image = image - lower

    return image


def read_dicom(path, window_center=None, window_width=None):
    """
    Reads a DICOM file and converts it to a numpy array, optionally applying windowing.

    Args:
        path (str): Path to the .dcm file.
        window_center (float, optional): Window center for bone windowing.
        window_width (float, optional): Window width for bone windowing.

    Returns:
        np.ndarray: The image array (float32). Returns zeros if file read fails.
    """
    if not os.path.exists(path):
        # Return a blank image of standard size if file is missing
        return np.zeros(
            (Config.ORIGINAL_IMAGE_SIZE, Config.ORIGINAL_IMAGE_SIZE), dtype=np.float32
        )

    try:
        ds = pydicom.dcmread(path)

        # Get pixel array
        # pydicom handles compression if appropriate libraries are installed
        image = ds.pixel_array.astype(np.float32)

        # Convert to Hounsfield Units (HU)
        slope = getattr(ds, "RescaleSlope", 1.0)
        intercept = getattr(ds, "RescaleIntercept", 0.0)

        image = image * slope + intercept

        # Apply windowing if parameters are provided
        if window_center is not None and window_width is not None:
            image = apply_windowing(image, window_center, window_width)

        return image.astype(np.float32)

    except Exception as e:
        # Fallback for corrupted files or read errors
        return np.zeros(
            (Config.ORIGINAL_IMAGE_SIZE, Config.ORIGINAL_IMAGE_SIZE), dtype=np.float32
        )


def get_competition_weights(device="cpu"):
    """
    Returns the weights for the competition metric.
    Weights are assigned based on the task description:
    - C1 to C7: Weight 1.0
    - patient_overall: Weight 7.0 (Weighted more highly)

    Args:
        device (str): Device to place the tensor on.

    Returns:
        torch.Tensor: Weights tensor of shape (8,).
    """
    # Columns correspond to: [C1, C2, C3, C4, C5, C6, C7, patient_overall]
    weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 7.0], device=device)
    return weights


def competition_loss(y_pred, y_true):
    """
    Calculates the weighted multi-label logarithmic loss.

    L_ij = -w_j * [y_ij * log(p_ij) + (1 - y_ij) * log(1 - p_ij)]
    Loss is averaged across all rows (all predictions in the batch).

    Args:
        y_pred (torch.Tensor): Predicted probabilities of shape (Batch, 8).
        y_true (torch.Tensor): Ground truth labels of shape (Batch, 8).

    Returns:
        torch.Tensor: Scalar loss value.
    """
    # Clamp predictions to prevent log(0)
    epsilon = 1e-7
    y_pred = torch.clamp(y_pred, epsilon, 1.0 - epsilon)

    device = y_pred.device
    weights = get_competition_weights(device)

    # Calculate Binary Cross Entropy for each element
    # reduction='none' retains the shape (Batch, 8)
    bce = F.binary_cross_entropy(y_pred, y_true.float(), reduction="none")

    # Apply weights to each class
    weighted_bce = bce * weights

    # Average across all entries (Batch * 8 rows)
    return weighted_bce.mean()


def save_checkpoint(model, optimizer, epoch, best_metric, path):
    """
    Saves the model checkpoint to the specified path.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer state.
        epoch (int): Current epoch number.
        best_metric (float): Best validation metric so far.
        path (str): Destination file path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "best_metric": best_metric,
    }
    torch.save(state, path)


def load_checkpoint(model, optimizer, path, device="cpu"):
    """
    Loads a model checkpoint from the specified path.

    Args:
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer): The optimizer to load state into.
        path (str): Path to the checkpoint file.
        device (str): Device to map the location to.

    Returns:
        tuple: (model, optimizer, epoch, best_metric)
    """
    if not os.path.exists(path):
        # Return defaults if no checkpoint exists
        return model, optimizer, 0, float("inf")

    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and checkpoint["optimizer_state_dict"] is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    best_metric = checkpoint.get("best_metric", float("inf"))

    return model, optimizer, epoch, best_metric
