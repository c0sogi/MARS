import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed: int = Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def dice_score_batch(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
):
    """
    Calculates the Global Dice coefficient over a flattened batch tensor.
    This treats the entire batch as a single volume for metric calculation,
    aligning with the 'Global Dice' definition where X and Y are sets of
    pixels over the dataset.

    Args:
        y_pred (torch.Tensor): Predicted probabilities or logits.
                               Shape (B, C, H, W) or (B, H, W).
        y_true (torch.Tensor): Ground truth binary masks.
                               Shape (B, C, H, W) or (B, H, W).
        threshold (float): Threshold to convert probabilities to binary mask.
        smooth (float): Smoothing factor to avoid division by zero.

    Returns:
        torch.Tensor: Scalar tensor containing the Dice score.
    """
    # Ensure targets are on the correct device and float type
    y_true = y_true.to(y_pred.device).float()

    # Flatten the tensors to compute the metric globally over the batch
    y_pred_flat = y_pred.view(-1)
    y_true_flat = y_true.view(-1)

    # Apply threshold to get binary predictions
    y_pred_bin = (y_pred_flat > threshold).float()

    # Calculate intersection and cardinality
    intersection = (y_pred_bin * y_true_flat).sum()
    cardinality_pred = y_pred_bin.sum()
    cardinality_true = y_true_flat.sum()

    # Compute Dice coefficient
    dice = (2.0 * intersection) / (cardinality_pred + cardinality_true + smooth)

    return dice


def rle_encode(mask: np.ndarray):
    """
    Converts a binary mask into Run-Length Encoding (RLE) format.

    The format is a space-delimited list of pairs: 'start length'.
    Pixels are numbered from top to bottom, then left to right (1-based indexing).

    Args:
        mask (np.ndarray): Binary mask of shape (H, W).
                           0 indicates background, 1 indicates object.

    Returns:
        str: RLE string (e.g., '1 3 10 5').
    """
    # Flatten column-wise (Fortran-style) to match the top-to-bottom, left-to-right order
    pixels = mask.flatten(order="F")

    # Prepend and append 0 to detect runs starting at index 0 or ending at the last index
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where the value changes (0->1 or 1->0)
    # runs[i] will be the index in the padded array.
    # Because we padded with one 0 at the start, these indices correspond
    # to 1-based indices in the original flattened array.
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # The 'runs' array contains alternating start and end indices.
    # We want lengths, so we subtract starts from ends.
    # runs[1::2] are the end indices (exclusive)
    # runs[::2] are the start indices (inclusive)
    runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)
