import os
import sys
import logging
import numpy as np
import torch
from library.config import Config


def set_seed(seed=None):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Wraps the Config.set_seed method to ensure consistency with the global configuration.

    Args:
        seed (int, optional): Specific seed to set. If None, uses Config.SEED.
    """
    if seed is not None:
        Config.SEED = seed
    Config.set_seed()


def rle_encode(mask):
    """
    Encodes a binary mask using Run-Length Encoding (RLE) formatted for submission.
    The pixels are numbered from top to bottom, then left to right (Column-Major/Fortran order).

    Args:
        mask (np.ndarray): Binary mask of shape (H, W) or (H, W, 1).
                           Values should be 0 (background) or 1 (contrail).

    Returns:
        str: Space-delimited list of pairs (start, length). Returns '-' if the mask is empty.
    """
    # Flatten in column-major order (Fortran-style) as per task description
    pixels = mask.flatten(order="F")

    # If no pixels are masked, return the specific empty indicator
    if np.sum(pixels) == 0:
        return "-"

    # Prepend and append 0 to detect transitions
    # This allows finding starts (0->1) and ends (1->0) efficiently
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where the value changes
    # np.where returns indices in the padded array
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # The 'runs' array now contains [start1, end1, start2, end2, ...]
    # We need [start1, length1, start2, length2, ...]
    # Length = end - start
    runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)


def dice_coef(y_pred, y_true, smooth=1e-6):
    """
    Computes the Dice Coefficient.
    Can be used as a metric (binary inputs) or a loss component (probability inputs).

    Args:
        y_pred (torch.Tensor or np.ndarray): Predicted map or probabilities.
        y_true (torch.Tensor or np.ndarray): Ground truth mask.
        smooth (float): Smoothing factor to avoid division by zero.

    Returns:
        torch.Tensor or float: The calculated Dice coefficient.
    """
    # Convert numpy arrays to tensors if necessary
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.from_numpy(y_pred)
    if isinstance(y_true, np.ndarray):
        y_true = torch.from_numpy(y_true)

    # Flatten the tensors
    y_pred_f = y_pred.reshape(-1)
    y_true_f = y_true.reshape(-1)

    intersection = (y_pred_f * y_true_f).sum()
    union = y_pred_f.sum() + y_true_f.sum()

    return (2.0 * intersection + smooth) / (union + smooth)


def global_dice_score(y_pred, y_true):
    """
    Computes the Global Dice Coefficient.
    Unlike standard Dice which might average over samples, this metric aggregates
    intersection and union over the entire set X and Y before division.

    Formula: 2 * |X n Y| / (|X| + |Y|)

    Args:
        y_pred (torch.Tensor or np.ndarray): Predicted binary maps for the dataset.
        y_true (torch.Tensor or np.ndarray): Ground truth binary maps for the dataset.

    Returns:
        float: The Global Dice score.
    """
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.from_numpy(y_pred)
    if isinstance(y_true, np.ndarray):
        y_true = torch.from_numpy(y_true)

    y_pred_f = y_pred.reshape(-1)
    y_true_f = y_true.reshape(-1)

    intersection = (y_pred_f * y_true_f).sum()
    union = y_pred_f.sum() + y_true_f.sum()

    # Handle the case where both prediction and ground truth are completely empty
    if union == 0:
        return 1.0

    score = (2.0 * intersection) / union
    return score.item()


def get_logger(name, log_file=None):
    """
    Initializes a logger that outputs to both console and a file.

    Args:
        name (str): Name of the logger.
        log_file (str, optional): Path to the log file.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Clear existing handlers to prevent duplicate logging
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Stream Handler (Console)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    # File Handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
