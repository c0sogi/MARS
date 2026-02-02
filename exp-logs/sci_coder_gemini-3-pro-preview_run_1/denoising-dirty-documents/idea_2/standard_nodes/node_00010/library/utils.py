import os
import numpy as np
import torch
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets random seeds for reproducibility by delegating to Config.seed_everything.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    Config.seed_everything(seed)


def calculate_rmse(y_true, y_pred):
    """
    Calculates the Root Mean Squared Error (RMSE) between true and predicted values.
    Handles both NumPy arrays and PyTorch tensors.

    Args:
        y_true: Ground truth values (np.ndarray or torch.Tensor).
        y_pred: Predicted values (np.ndarray or torch.Tensor).

    Returns:
        float: The calculated RMSE.
    """
    # Convert torch tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Calculate MSE then RMSE
    mse = np.mean((y_true - y_pred) ** 2)
    return np.sqrt(mse)


def pad_image(image, factor=32):
    """
    Pads an image so that its height and width are divisible by a given factor.
    Uses reflection padding on the bottom and right edges to minimize artifacts.

    Args:
        image: Input image as a NumPy array (H, W) or (H, W, C).
        factor: The divisor that dimensions must align with (default 32).

    Returns:
        np.ndarray: The padded image.
    """
    h, w = image.shape[:2]

    # Calculate required padding to make dimensions divisible by factor
    h_pad = (factor - h % factor) % factor
    w_pad = (factor - w % factor) % factor

    # Construct pad_width: [(top, bottom), (left, right), (optional_channels)]
    # We pad only bottom and right to simplify unpadding (slicing)
    pad_width = [(0, h_pad), (0, w_pad)]

    # Handle channels if present (e.g., shape (H, W, C))
    if image.ndim == 3:
        pad_width.append((0, 0))

    # Use reflection padding to extend texture naturally
    padded_image = np.pad(image, pad_width, mode="reflect")

    return padded_image


def unpad_image(padded_image, original_shape):
    """
    Crops a padded image back to its original dimensions.

    Args:
        padded_image: The image with padding added.
        original_shape: Tuple of (H, W) or (H, W, C) representing the target size.

    Returns:
        np.ndarray: The cropped image.
    """
    h, w = original_shape[:2]
    return padded_image[:h, :w]
