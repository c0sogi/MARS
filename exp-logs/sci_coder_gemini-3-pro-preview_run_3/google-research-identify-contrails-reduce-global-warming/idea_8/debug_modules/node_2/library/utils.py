import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def rle_encode(mask):
    """
    Encodes a binary mask using Run-Length Encoding (RLE).

    The format is a space-delimited list of pairs: 'start length'.
    Pixels are numbered from top to bottom, then left to right (Fortran-style flattening).

    Args:
        mask (np.ndarray): Binary mask of shape (H, W).

    Returns:
        str: RLE string or '-' if the mask is empty.
    """
    # Flatten column-wise (Fortran-style) as per competition spec
    pixels = mask.flatten(order="F")

    # Check if mask is empty
    if np.sum(pixels) == 0:
        return "-"

    # Pad with 0s to detect start/end of runs correctly at boundaries
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where values change (0->1 or 1->0)
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # Calculate lengths: end_index - start_index
    # The runs array currently holds [start1, end1, start2, end2, ...]
    # We want [start1, length1, start2, length2, ...]
    runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)


def dice_coef(y_pred, y_true, threshold=0.5, smooth=1e-6, from_logits=True):
    """
    Calculates the Global Dice Coefficient for a batch for validation monitoring.

    The metric is defined as 2 * |X n Y| / (|X| + |Y|), where X and Y are the
    sets of predicted and ground truth pixels respectively.

    Args:
        y_pred (torch.Tensor): Predicted logits or probabilities.
        y_true (torch.Tensor): Ground truth binary mask.
        threshold (float): Threshold for binarization.
        smooth (float): Smoothing factor to avoid division by zero.
        from_logits (bool): If True, applies sigmoid to y_pred before thresholding.

    Returns:
        float: The calculated Dice coefficient score.
    """
    if from_logits:
        y_pred = torch.sigmoid(y_pred)

    # Binarize predictions
    y_pred = (y_pred > threshold).float()
    y_true = y_true.float()

    # Flatten tensors to compute global dice over the entire batch
    # This treats the batch as a single large volume/set of pixels
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)

    intersection = (y_pred * y_true).sum()
    union = y_pred.sum() + y_true.sum()

    dice = (2.0 * intersection) / (union + smooth)

    return dice.item()
