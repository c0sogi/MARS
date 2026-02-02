import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed: int = 42):
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

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_rmse(y_true, y_pred):
    """
    Calculates the Root Mean Squared Error (RMSE) between true and predicted values.
    Supports both NumPy arrays and PyTorch tensors.

    Args:
        y_true: Ground truth values (Tensor or ndarray).
        y_pred: Predicted values (Tensor or ndarray).

    Returns:
        float: The RMSE value.
    """
    # Convert PyTorch tensors to NumPy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure float type for high precision calculation
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)

    mse = np.mean((y_true - y_pred) ** 2)
    return np.sqrt(mse)


def apply_tta(image: torch.Tensor, k: int) -> torch.Tensor:
    """
    Applies one of the 8 geometric transformations from the D4 symmetry group.
    Used for Geometric Self-Ensemble (Test Time Augmentation).

    Args:
        image: Tensor of shape (..., H, W).
        k: Integer from 0 to 7 indicating the transformation index.

    Returns:
        torch.Tensor: The transformed image.
    """
    # 0: Identity
    if k == 0:
        return image
    # 1: Rotate 90 degrees counter-clockwise
    elif k == 1:
        return torch.rot90(image, 1, (-2, -1))
    # 2: Rotate 180 degrees
    elif k == 2:
        return torch.rot90(image, 2, (-2, -1))
    # 3: Rotate 270 degrees counter-clockwise
    elif k == 3:
        return torch.rot90(image, 3, (-2, -1))
    # 4: Horizontal Flip
    elif k == 4:
        return torch.flip(image, [-1])
    # 5: Vertical Flip
    elif k == 5:
        return torch.flip(image, [-2])
    # 6: Transpose (Swap H and W)
    elif k == 6:
        return torch.transpose(image, -2, -1)
    # 7: Anti-Transpose (Rot90 + Vertical Flip)
    elif k == 7:
        # Rotate 90 then Flip Vertically
        return torch.flip(torch.rot90(image, 1, (-2, -1)), [-2])
    else:
        raise ValueError(f"Invalid TTA index k={k}. Must be 0-7.")


def reverse_tta(image: torch.Tensor, k: int) -> torch.Tensor:
    """
    Reverses the geometric transformation applied by apply_tta.

    Args:
        image: Tensor of shape (..., H, W).
        k: Integer from 0 to 7 indicating the transformation index to reverse.

    Returns:
        torch.Tensor: The image transformed back to original orientation.
    """
    # 0: Identity -> Identity
    if k == 0:
        return image
    # 1: Rot90 -> Rot270 (reverse is Rot-90)
    elif k == 1:
        return torch.rot90(image, 3, (-2, -1))
    # 2: Rot180 -> Rot180
    elif k == 2:
        return torch.rot90(image, 2, (-2, -1))
    # 3: Rot270 -> Rot90
    elif k == 3:
        return torch.rot90(image, 1, (-2, -1))
    # 4: HFlip -> HFlip (inverse of flip is flip)
    elif k == 4:
        return torch.flip(image, [-1])
    # 5: VFlip -> VFlip
    elif k == 5:
        return torch.flip(image, [-2])
    # 6: Transpose -> Transpose (inverse of transpose is transpose)
    elif k == 6:
        return torch.transpose(image, -2, -1)
    # 7: Anti-Transpose -> Anti-Transpose
    # Forward was: FlipV(Rot90(x))
    # Inverse is: Rot-90(FlipV(x)) -> Rot270(FlipV(x))
    elif k == 7:
        return torch.rot90(torch.flip(image, [-2]), 3, (-2, -1))
    else:
        raise ValueError(f"Invalid TTA index k={k}. Must be 0-7.")
