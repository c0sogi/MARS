import os
import random
import numpy as np
import torch
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.
    Configures cudnn for deterministic execution.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def dice_coefficient(y_pred, y_true, smooth=1e-6):
    """
    Computes the Dice coefficient for a batch of predictions.
    This calculates the global dice over the provided batch (sum of intersections / sum of unions),
    which aligns with the global metric definition when accumulated.

    Args:
        y_pred (torch.Tensor): Predicted probabilities or binary masks.
        y_true (torch.Tensor): Ground truth binary masks.
        smooth (float): Smoothing factor to avoid division by zero.

    Returns:
        float: The Dice coefficient.
    """
    # Flatten the tensors to treat the batch as a single set of pixels
    y_pred_f = y_pred.view(-1)
    y_true_f = y_true.view(-1)

    intersection = (y_pred_f * y_true_f).sum()
    union = y_pred_f.sum() + y_true_f.sum()

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.item()


def rle_encode(mask):
    """
    Run-length encoding for a binary mask.
    The pixels are numbered from top to bottom, then left to right.

    Args:
        mask (numpy.ndarray): Binary mask of shape (H, W).

    Returns:
        str: Space-delimited list of pairs 'start length' or '-' if empty.
    """
    # Flatten in column-major order (Fortran-style) as per task description
    pixels = mask.flatten(order="F")

    # Prepend and append 0 to detect start and end of runs
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where values change (0->1 or 1->0)
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # The 'runs' array now contains [start1, end1, start2, end2, ...]
    # We want [start1, length1, start2, length2, ...]
    # Calculate lengths: end - start
    runs[1::2] -= runs[::2]

    if len(runs) == 0:
        return "-"

    return " ".join(str(x) for x in runs)
