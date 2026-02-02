import numpy as np
import torch
import pandas as pd
from library.config import Config


def seed_everything(seed: int = Config.SEED):
    """
    Sets the random seed for reproducibility using the Config method.
    """
    Config.seed_everything(seed)


def rle_encode(img: np.ndarray) -> str:
    """
    Run-length encoding for a binary mask.
    Pixels are numbered from top to bottom, then left to right.

    Args:
        img (np.ndarray): Binary mask (0s and 1s).

    Returns:
        str: RLE string.
    """
    pixels = img.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle: str, shape: tuple) -> np.ndarray:
    """
    Decodes a run-length encoded string into a binary mask.

    Args:
        mask_rle (str): RLE string.
        shape (tuple): Target shape (height, width).

    Returns:
        np.ndarray: Binary mask.
    """
    if pd.isna(mask_rle) or mask_rle == "nan":
        return np.zeros(shape, dtype=np.uint8)

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order="F")


def dice_coef(y_pred, y_true, smooth: float = 1e-6):
    """
    Computes the Dice Coefficient.

    Args:
        y_pred (torch.Tensor or np.ndarray): Predictions (probabilities or binary).
        y_true (torch.Tensor or np.ndarray): Ground truth.
        smooth (float): Smoothing factor.

    Returns:
        float: Dice score.
    """
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.from_numpy(y_pred)
    if isinstance(y_true, np.ndarray):
        y_true = torch.from_numpy(y_true)

    # Flatten tensors
    y_pred = y_pred.view(-1).float()
    y_true = y_true.view(-1).float()

    intersection = (y_pred * y_true).sum()
    dice = (2.0 * intersection + smooth) / (y_pred.sum() + y_true.sum() + smooth)

    return dice.item()
