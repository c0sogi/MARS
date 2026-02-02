import numpy as np
import torch
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility by delegating to the Config class.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    Config.set_seed(seed)


def rle_encode(mask):
    """
    Encodes a binary mask using Run-Length Encoding (RLE) for submission.

    The mask is flattened in column-major order (top-to-bottom, then left-to-right).
    The output is a space-delimited list of pairs: 'start length'.
    Indices are 1-based.

    Args:
        mask (np.ndarray): Binary mask of shape (H, W). 1 indicates mask, 0 indicates background.

    Returns:
        str: RLE string (e.g., '1 3 10 5') or '-' if the mask is empty.
    """
    # Flatten in column-major order (Fortran-style) as per task description
    pixels = mask.flatten(order="F")

    # If the mask is empty, return '-'
    if np.sum(pixels) == 0:
        return "-"

    # Pad the pixels with 0 at start and end to detect all transitions
    # This handles runs starting at the first pixel or ending at the last pixel
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where the value changes (0->1 or 1->0)
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # 'runs' currently holds the start indices of value changes.
    # Because we padded with 0, the first change (if any) must be 0->1 (start of run).
    # The next change must be 1->0 (end of run).
    # We calculate lengths by subtracting start indices from end indices.
    # runs[1::2] are the end indices (exclusive in 0-based, but effectively end+1)
    # runs[::2] are the start indices.
    runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)


def dice_score_batch(preds, targets, smooth=1e-6):
    """
    Calculates the Global Dice Coefficient for a set of predictions.

    This function computes the Dice score over the entire input tensors, treating them
    as a single volume (Batch-Level Dice). This approximates the Global Dice metric
    used in evaluation when applied to the full validation set.

    Formula: 2 * |X n Y| / (|X| + |Y|)

    Args:
        preds (torch.Tensor or np.ndarray): Predicted binary masks (0 or 1).
        targets (torch.Tensor or np.ndarray): Ground truth binary masks (0 or 1).
        smooth (float): Small epsilon to avoid division by zero.

    Returns:
        float: The Global Dice score.
    """
    # Convert NumPy arrays to PyTorch tensors if necessary
    if isinstance(preds, np.ndarray):
        preds = torch.from_numpy(preds)
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets)

    # Flatten inputs to 1D vectors to compute global intersection and cardinality
    preds_flat = preds.reshape(-1).float()
    targets_flat = targets.reshape(-1).float()

    # Calculate Intersection: |X n Y|
    intersection = (preds_flat * targets_flat).sum()

    # Calculate Cardinality: |X| + |Y|
    union = preds_flat.sum() + targets_flat.sum()

    # Compute Dice
    dice = (2.0 * intersection) / (union + smooth)

    return dice.item()
