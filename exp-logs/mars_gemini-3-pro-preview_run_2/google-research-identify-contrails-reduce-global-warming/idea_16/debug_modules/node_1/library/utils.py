import os
import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set environment variable for hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


class AverageMeter:
    """
    Computes and stores the average and current value of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        Update the meter with a new value.

        Args:
            val (float): The current value.
            n (int): The number of samples associated with this value.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def dice_coef(y_pred, y_true, smooth=1e-6):
    """
    Calculates the Dice Coefficient.

    Formula: 2 * |X n Y| / (|X| + |Y|)

    This function flattens the inputs, effectively computing the metric
    globally over the batch (or image) provided.

    Args:
        y_pred (torch.Tensor): Predicted output. Can be probabilities (for soft dice)
                               or binary (for hard metric).
        y_true (torch.Tensor): Ground truth binary mask.
        smooth (float): Smoothing factor to avoid division by zero.

    Returns:
        float: The calculated Dice coefficient.
    """
    # Flatten the tensors to compute over the entire volume
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()

    intersection = torch.sum(y_true_f * y_pred_f)
    union = torch.sum(y_true_f) + torch.sum(y_pred_f)

    return (2.0 * intersection + smooth) / (union + smooth)


def rle_encode(mask):
    """
    Run-length encoding for a binary mask.

    The metric specifies: "The pixels are numbered from top to bottom, then left to right".
    This corresponds to column-major flattening (Fortran style).

    Args:
        mask (np.ndarray): Binary mask of shape (H, W). 1 indicates mask, 0 indicates background.

    Returns:
        str: Run-length encoded string in 'start length' pairs, space delimited.
             Returns '-' if the mask is empty.
    """
    # Flatten column-major (Fortran style) as per task specification
    pixels = mask.flatten(order="F")

    # Handle empty prediction
    if np.sum(pixels) == 0:
        return "-"

    # Pad with 0s at start and end to detect runs at boundaries
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where value changes (0->1 or 1->0)
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # Calculate lengths: indices of ends (1->0) - indices of starts (0->1)
    # runs[0] is start, runs[1] is end of first run, etc.
    runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)
