import os
import random
import numpy as np
import torch
import torch.nn.functional as F


def seed_everything(seed: int):
    """
    Seeds all random number generators for reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pad_to_multiple(x: torch.Tensor, divisor: int = 16):
    """
    Pads a tensor (C, H, W) or (B, C, H, W) so that H and W are multiples of divisor.
    Uses reflection padding to minimize boundary artifacts.

    Args:
        x: Input tensor.
        divisor: The factor H and W must be divisible by.

    Returns:
        padded_x: The padded tensor.
        padding: Tuple (pad_l, pad_r, pad_t, pad_b) used for unpadding.
    """
    # Handle dimensions
    if x.dim() == 3:
        h, w = x.shape[1], x.shape[2]
    elif x.dim() == 4:
        h, w = x.shape[2], x.shape[3]
    else:
        raise ValueError(f"Unsupported tensor dimension: {x.dim()}. Expected 3 or 4.")

    pad_h = (divisor - h % divisor) % divisor
    pad_w = (divisor - w % divisor) % divisor

    # F.pad expects (left, right, top, bottom)
    # We pad right and bottom to keep coordinates aligned
    padding = (0, pad_w, 0, pad_h)

    # Use reflect padding to avoid boundary artifacts
    padded_x = F.pad(x, padding, mode="reflect")

    return padded_x, padding


def unpad(x: torch.Tensor, padding: tuple):
    """
    Removes padding from a tensor.

    Args:
        x: Padded tensor.
        padding: Tuple (pad_l, pad_r, pad_t, pad_b) returned by pad_to_multiple.

    Returns:
        Unpadded tensor.
    """
    pad_l, pad_r, pad_t, pad_b = padding

    h, w = x.shape[-2:]

    # Calculate end indices
    # If pad is 0, we want to go to the end (h or w)
    end_h = h - pad_b
    end_w = w - pad_r

    return x[..., pad_t:end_h, pad_l:end_w]


def rmse_loss(pred: torch.Tensor, target: torch.Tensor):
    """
    Calculates Root Mean Squared Error between prediction and target.
    """
    return torch.sqrt(F.mse_loss(pred, target))


def get_tta_transforms():
    """
    Returns a list of 8 geometric transformations (D4 group) for Test-Time Augmentation.
    The transformations operate on tensors of shape (..., H, W).
    """
    return [
        lambda x: x,  # Original
        lambda x: torch.rot90(x, 1, [2, 3]),  # Rot90
        lambda x: torch.rot90(x, 2, [2, 3]),  # Rot180
        lambda x: torch.rot90(x, 3, [2, 3]),  # Rot270
        lambda x: torch.flip(x, [3]),  # HFlip
        lambda x: torch.rot90(torch.flip(x, [3]), 1, [2, 3]),  # HFlip + Rot90
        lambda x: torch.rot90(torch.flip(x, [3]), 2, [2, 3]),  # HFlip + Rot180
        lambda x: torch.rot90(torch.flip(x, [3]), 3, [2, 3]),  # HFlip + Rot270
    ]


def inverse_tta_transforms():
    """
    Returns a list of inverse transformations corresponding to the list from get_tta_transforms.
    Used to restore the original orientation of the prediction.
    """
    return [
        lambda x: x,  # Inv Original
        lambda x: torch.rot90(x, 3, [2, 3]),  # Inv Rot90 (is Rot270)
        lambda x: torch.rot90(x, 2, [2, 3]),  # Inv Rot180 (is Rot180)
        lambda x: torch.rot90(x, 1, [2, 3]),  # Inv Rot270 (is Rot90)
        lambda x: torch.flip(x, [3]),  # Inv HFlip
        lambda x: torch.flip(torch.rot90(x, 3, [2, 3]), [3]),  # Inv (HFlip + Rot90)
        lambda x: torch.flip(torch.rot90(x, 2, [2, 3]), [3]),  # Inv (HFlip + Rot180)
        lambda x: torch.flip(torch.rot90(x, 1, [2, 3]), [3]),  # Inv (HFlip + Rot270)
    ]
