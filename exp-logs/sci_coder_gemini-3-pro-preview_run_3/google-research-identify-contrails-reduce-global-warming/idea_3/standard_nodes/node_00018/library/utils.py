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
    torch.cuda.manual_seed(seed)
    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


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


def rle_encode(mask):
    """
    Run-Length Encode a binary mask.

    The mask is flattened in column-major order (top-to-bottom, then left-to-right).
    1-based indexing is used.

    Args:
        mask (np.ndarray): Binary mask of shape (H, W).

    Returns:
        str: Space-delimited list of pairs (start, length) or '-' if empty.
    """
    # Flatten column-major as per competition spec
    pixels = mask.flatten(order="F")

    # We need to find runs of 1s.
    # Concatenate 0 at start and end to detect transitions at boundaries
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where values change (0->1 or 1->0)
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # If no runs found (all zeros), return '-'
    if len(runs) == 0:
        return "-"

    # runs array structure: [start_1, end_1, start_2, end_2, ...]
    # We need lengths: length = end - start
    # The start indices in `runs` are already 1-based relative to the original flattened array
    # because we prepended a 0.

    # Calculate lengths in place: odd indices (ends) minus even indices (starts)
    runs[1::2] -= runs[::2]

    # Format as space-separated string
    return " ".join(str(x) for x in runs)


def dice_score(y_pred, y_true, smooth=1e-6, threshold=0.5):
    """
    Compute the Dice coefficient.

    Args:
        y_pred (torch.Tensor): Predicted probabilities or logits.
        y_true (torch.Tensor): Ground truth masks.
        smooth (float): Smoothing factor to prevent division by zero.
        threshold (float, optional): Threshold to binarize predictions.
                                     If None, computes soft Dice.

    Returns:
        float: The Dice score.
    """
    # Ensure tensors are on the same device
    y_true = y_true.to(y_pred.device)

    # Flatten the tensors
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)

    # Apply threshold if provided (Hard Dice)
    if threshold is not None:
        y_pred = (y_pred > threshold).float()

    # Calculate Intersection and Union
    intersection = (y_pred * y_true).sum()
    union = y_pred.sum() + y_true.sum()

    # Compute Dice
    score = (2.0 * intersection) / (union + smooth)

    return score.item()
