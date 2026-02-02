import os
import random
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import SEED


# ==========================================
# Reproducibility
# ==========================================
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed()


# ==========================================
# Data Processing Utilities
# ==========================================
def normalize_range(data, min_val, max_val):
    """
    Scales data linearly to [0, 1] based on min_val and max_val.
    Clips values outside the range to ensure valid bounds.

    Args:
        data (np.ndarray): Input array (e.g., band difference).
        min_val (float): Minimum physical value.
        max_val (float): Maximum physical value.

    Returns:
        np.ndarray: Normalized and clipped data.
    """
    data = (data - min_val) / (max_val - min_val)
    return np.clip(data, 0, 1)


def rle_encode(mask):
    """
    Run-length encoding for binary masks.
    The pixels are numbered from top to bottom, then left to right (Column-Major).
    1 is pixel (1,1).

    Args:
        mask (np.ndarray): Binary mask of shape (H, W).

    Returns:
        str: Space-delimited list of pairs (start length) or '-' if empty.
    """
    # Flatten column-major (Fortran-style) to match top-to-bottom, left-to-right indexing
    pixels = mask.flatten(order="F")

    # Check if mask is empty
    if np.sum(pixels) == 0:
        return "-"

    # Pad with zeros to detect start/end of runs
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where value changes
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # Calculate lengths: end_index - start_index
    runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)


def get_transforms(data="train"):
    """
    Returns the Albumentations transform pipeline.
    Implements strict Affine transformations (Rotate, Scale, Shift, Flip).
    Explicitly excludes elastic, grid, or optical distortions.

    Args:
        data (str): 'train', 'valid', or 'test'.

    Returns:
        A.Compose: Composed transforms.
    """
    if data == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                # Affine transforms: Shift, Scale, Rotate
                A.ShiftScaleRotate(
                    shift_limit=0.05,
                    scale_limit=0.05,
                    rotate_limit=30,
                    p=0.5,
                    border_mode=0,  # Constant padding with 0
                    value=0,
                ),
                ToTensorV2(),
            ]
        )
    else:
        # Validation and Test: Only convert to Tensor
        return A.Compose([ToTensorV2()])


# ==========================================
# Evaluation Metrics
# ==========================================
def dice_coeff(pred, target, smooth=1e-6):
    """
    Calculates the Dice Coefficient (F1 Score) for a batch or single image.
    Formula: 2 * |X n Y| / (|X| + |Y|)

    Args:
        pred (torch.Tensor): Predicted probabilities or binary mask.
        target (torch.Tensor): Ground truth mask.
        smooth (float): Smoothing factor to prevent division by zero.

    Returns:
        torch.Tensor: Scalar Dice score.
    """
    pred = pred.view(-1)
    target = target.view(-1)

    intersection = (pred * target).sum()
    return (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
