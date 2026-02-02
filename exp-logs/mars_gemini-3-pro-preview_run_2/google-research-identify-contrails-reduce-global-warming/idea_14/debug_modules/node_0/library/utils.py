import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed: int = Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def rle_encode(mask, threshold=0.5):
    """
    Encodes a binary mask using Run-Length Encoding (RLE).
    The pixels are numbered from top to bottom, then left to right.

    Args:
        mask (np.ndarray or torch.Tensor): Binary mask of shape (H, W).
                                           Values should be 0 or 1, or probabilities.
        threshold (float): Threshold to binarize probabilities if necessary.

    Returns:
        str: Space-delimited list of pairs (start, length) or '-' if empty.
    """
    # Convert tensor to numpy if needed
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    # Squeeze channel dim if present (e.g., 1xHxW -> HxW)
    if mask.ndim == 3:
        mask = mask.squeeze(0)

    # Binarize
    if mask.dtype == float or mask.dtype == np.float32 or mask.dtype == np.float64:
        mask = (mask > threshold).astype(np.uint8)
    else:
        mask = mask.astype(np.uint8)

    # Flatten column-wise (Fortran-style) as per task description
    pixels = mask.flatten(order="F")

    # Prepend and append 0 to detect changes at boundaries
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where values change
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # Calculate lengths (end - start)
    runs[1::2] -= runs[::2]

    encoded = " ".join(str(x) for x in runs)

    return encoded if encoded else "-"


def dice_coef_metric(y_pred, y_true, threshold=Config.THRESHOLD, epsilon=1e-6):
    """
    Computes the Dice Coefficient between prediction and ground truth.

    Args:
        y_pred (torch.Tensor or np.ndarray): Predicted probabilities or logits.
        y_true (torch.Tensor or np.ndarray): Ground truth binary mask.
        threshold (float): Threshold for binarizing predictions.
        epsilon (float): Small constant to prevent division by zero.

    Returns:
        float: Dice coefficient.
    """
    if isinstance(y_pred, torch.Tensor):
        # Apply sigmoid if values are logits (heuristically check range)
        if y_pred.min() < 0 or y_pred.max() > 1:
            y_pred = torch.sigmoid(y_pred)

        y_pred = (y_pred > threshold).float()
        y_true = y_true.float()

        # Flatten to calculate intersection over the volume/image
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)

        intersection = (y_pred * y_true).sum()
        union = y_pred.sum() + y_true.sum()

        dice = (2.0 * intersection) / (union + epsilon)
        return dice.item()

    elif isinstance(y_pred, np.ndarray):
        if y_pred.min() < 0 or y_pred.max() > 1:
            y_pred = 1 / (1 + np.exp(-y_pred))

        y_pred = (y_pred > threshold).astype(np.float32)
        y_true = y_true.astype(np.float32)

        y_pred = y_pred.flatten()
        y_true = y_true.flatten()

        intersection = np.sum(y_pred * y_true)
        union = np.sum(y_pred) + np.sum(y_true)

        dice = (2.0 * intersection) / (union + epsilon)
        return dice

    return 0.0


def compute_intersection_union(y_pred, y_true, threshold=Config.THRESHOLD):
    """
    Computes intersection and union sum for Global Dice calculation.
    Useful for accumulating stats over an entire dataset.

    Args:
        y_pred (torch.Tensor or np.ndarray): Predicted probabilities or logits.
        y_true (torch.Tensor or np.ndarray): Ground truth binary mask.
        threshold (float): Threshold for binarizing predictions.

    Returns:
        tuple: (intersection, cardinality) where cardinality = |X| + |Y|
    """
    if isinstance(y_pred, torch.Tensor):
        if y_pred.min() < 0 or y_pred.max() > 1:
            y_pred = torch.sigmoid(y_pred)
        y_pred = (y_pred > threshold).float()
        y_true = y_true.float()

        intersection = (y_pred * y_true).sum().item()
        cardinality = (y_pred.sum() + y_true.sum()).item()
        return intersection, cardinality
    else:
        # Numpy
        if y_pred.min() < 0 or y_pred.max() > 1:
            y_pred = 1 / (1 + np.exp(-y_pred))
        y_pred = (y_pred > threshold).astype(np.float32)
        y_true = y_true.astype(np.float32)

        intersection = np.sum(y_pred * y_true)
        cardinality = np.sum(y_pred) + np.sum(y_true)
        return intersection, cardinality


class AverageMeter:
    """Computes and stores the average and current value."""

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
