import os
import random
import numpy as np
import torch
import torch.nn.functional as F


def set_seed(seed: int = 42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to set.
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


def calculate_rmse(y_true, y_pred):
    """
    Calculates the Root Mean Squared Error (RMSE) between true and predicted values.

    Args:
        y_true (torch.Tensor or np.ndarray): Ground truth tensor/array.
        y_pred (torch.Tensor or np.ndarray): Predicted tensor/array.

    Returns:
        float: The RMSE value.
    """
    # Ensure inputs are tensors
    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true)
    if not isinstance(y_pred, torch.Tensor):
        y_pred = torch.tensor(y_pred)

    # Calculate MSE
    mse = F.mse_loss(y_pred, y_true)

    # Return RMSE as a standard float
    return torch.sqrt(mse).item()


def pad_image(image, divisor=32, mode="reflect"):
    """
    Pads an image tensor so that its spatial dimensions are divisible by the given divisor.
    This is useful for U-Net architectures which require dimensions to be multiples of 2^n.

    Args:
        image (torch.Tensor): Input image tensor of shape (C, H, W) or (B, C, H, W).
        divisor (int): The number that the dimensions should be divisible by.
        mode (str): Padding mode (e.g., 'reflect', 'constant', 'replicate').

    Returns:
        torch.Tensor: Padded image tensor.
    """
    # Get current dimensions
    if image.dim() == 4:
        h, w = image.shape[2], image.shape[3]
    elif image.dim() == 3:
        h, w = image.shape[1], image.shape[2]
    else:
        raise ValueError(
            f"Unsupported image dimension: {image.dim()}. Expected 3 or 4."
        )

    # Calculate padding required
    pad_h = (divisor - h % divisor) % divisor
    pad_w = (divisor - w % divisor) % divisor

    # If no padding needed, return original
    if pad_h == 0 and pad_w == 0:
        return image

    # Pad format for F.pad is (left, right, top, bottom)
    # We pad right and bottom
    padding = (0, pad_w, 0, pad_h)

    padded_image = F.pad(image, padding, mode=mode)
    return padded_image


def unpad_image(image, original_shape):
    """
    Crops a padded image tensor back to its original shape.

    Args:
        image (torch.Tensor): Padded image tensor.
        original_shape (tuple): The original (Height, Width) before padding.

    Returns:
        torch.Tensor: Unpadded/Cropped image tensor.
    """
    h_orig, w_orig = original_shape

    # Slice based on dimensions
    if image.dim() == 4:
        return image[:, :, :h_orig, :w_orig]
    elif image.dim() == 3:
        return image[:, :h_orig, :w_orig]
    else:
        raise ValueError(
            f"Unsupported image dimension: {image.dim()}. Expected 3 or 4."
        )
