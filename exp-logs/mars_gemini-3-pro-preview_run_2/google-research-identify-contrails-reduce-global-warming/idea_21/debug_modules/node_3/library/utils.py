import numpy as np
import torch
from library.config import seed_everything


def rle_encode(mask):
    """
    Encodes a binary mask using Run-Length Encoding (RLE).

    The pixels are numbered from top to bottom, then left to right.

    Args:
        mask (np.ndarray): Binary mask of shape (Height, Width).
                           1 indicates mask, 0 indicates background.

    Returns:
        str: Space-delimited list of pairs (start, length) or '-' if empty.
    """
    # Flatten column-wise (Fortran-style) to match the top-to-bottom, left-to-right indexing
    pixels = mask.flatten(order="F")

    # Prepend and append 0 to detect transitions at the start and end
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where values change (0->1 or 1->0)
    changes = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # If no changes, the mask is empty
    if len(changes) == 0:
        return "-"

    # changes[0::2] are start indices (inclusive)
    # changes[1::2] are end indices (exclusive)
    # Calculate lengths: length = end - start
    changes[1::2] -= changes[::2]

    return " ".join(str(x) for x in changes)


def dice_coef(y_true, y_pred, smooth=1e-6):
    """
    Computes the Global Dice Coefficient.

    Formula: 2 * |X n Y| / (|X| + |Y|)
    where X is the set of predicted pixels and Y is the ground truth.

    Args:
        y_true (torch.Tensor or np.ndarray): Ground truth binary masks.
        y_pred (torch.Tensor or np.ndarray): Predicted binary masks.
        smooth (float): Small epsilon to avoid division by zero.

    Returns:
        float: The Global Dice coefficient.
    """
    # Convert tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Flatten inputs to compute global stats
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()

    intersection = np.sum(y_true_f * y_pred_f)
    cardinality = np.sum(y_true_f) + np.sum(y_pred_f)

    # Handle the case where both sets are empty (perfect match of background)
    if cardinality == 0:
        return 1.0

    return (2.0 * intersection) / (cardinality + smooth)
