import os
import numpy as np
import torch
from library.config import Config


def set_seed(seed=None):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Delegates to the implementation in Config to ensure consistency.

    Args:
        seed (int, optional): The seed value to set. If None, uses Config.SEED.
    """
    if seed is None:
        seed = Config.SEED
    Config.set_seed(seed)


def calculate_rmse(y_true, y_pred):
    """
    Calculates the Root Mean Squared Error (RMSE) between true and predicted values.
    Handles both numpy arrays and torch tensors.

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth values.
        y_pred (np.ndarray or torch.Tensor): Predicted values.

    Returns:
        float: The RMSE value.
    """
    # Convert torch tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure inputs are flattened for pixel-wise comparison
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    # Calculate MSE then RMSE
    mse = np.mean((y_true_flat - y_pred_flat) ** 2)
    rmse = np.sqrt(mse)

    return rmse


def normalize_image(image):
    """
    Normalizes an image array from [0, 255] to [0, 1].

    Args:
        image (np.ndarray): Input image array (uint8 or similar).

    Returns:
        np.ndarray: Normalized image array (float32).
    """
    return image.astype(np.float32) / 255.0


def denormalize_image(image):
    """
    Denormalizes an image array from [0, 1] to [0, 255] and converts to uint8.

    Args:
        image (np.ndarray): Input image array (float).

    Returns:
        np.ndarray: Denormalized image array (uint8).
    """
    # Clip to ensure valid range before scaling
    image = np.clip(image, 0, 1)
    return (image * 255.0).astype(np.uint8)


def get_device():
    """
    Returns the appropriate PyTorch device (CUDA if available, else CPU).

    Returns:
        torch.device: The device to use for computation.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def count_parameters(model):
    """
    Counts the number of trainable parameters in a PyTorch model.

    Args:
        model (torch.nn.Module): The model to inspect.

    Returns:
        int: The number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_metric(name, value):
    """
    Prints a metric name and value with full precision.

    Args:
        name (str): The name of the metric.
        value (float): The value of the metric.
    """
    print(f"{name}: {value}")
