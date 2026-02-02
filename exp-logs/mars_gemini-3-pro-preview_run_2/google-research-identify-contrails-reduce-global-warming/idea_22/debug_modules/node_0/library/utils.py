import os
import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
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


def rle_encode(mask):
    """
    Encodes a binary mask using Run-Length Encoding (RLE) for submission.

    The metric checks that the pairs are sorted, positive, and the decoded pixel
    values are not duplicated. The pixels are numbered from top to bottom,
    then left to right: 1 is pixel (1,1), 2 is pixel (2,1), etc.

    Args:
        mask (np.ndarray): Binary mask of shape (Height, Width).
                           1 indicates mask, 0 indicates background.

    Returns:
        str: Space-delimited list of pairs 'start length' or '-' if empty.
    """
    # Flatten column-wise (Fortran-style) as per requirements
    pixels = mask.flatten(order="F")

    # If mask is completely empty, return '-'
    if np.sum(pixels) == 0:
        return "-"

    # Pad with 0 at start and end to detect transitions
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where value changes
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # runs[0::2] are starts, runs[1::2] are ends
    # Lengths = ends - starts
    runs[1::2] -= runs[0::2]

    return " ".join(str(x) for x in runs)


def dice_score(preds, targets, smooth=1e-6):
    """
    Computes the Dice coefficient.

    Formula: 2 * |X \cap Y| / (|X| + |Y|)

    Args:
        preds (torch.Tensor or np.ndarray): Predicted binary mask or probabilities.
        targets (torch.Tensor or np.ndarray): Ground truth binary mask.
        smooth (float): Smoothing factor to avoid division by zero.

    Returns:
        float: The Dice coefficient.
    """
    # Convert to numpy if tensors
    if torch.is_tensor(preds):
        preds = preds.detach().cpu().numpy()
    if torch.is_tensor(targets):
        targets = targets.detach().cpu().numpy()

    # Flatten
    preds_flat = preds.flatten()
    targets_flat = targets.flatten()

    intersection = np.sum(preds_flat * targets_flat)
    union = np.sum(preds_flat) + np.sum(targets_flat)

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and metrics during training.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
