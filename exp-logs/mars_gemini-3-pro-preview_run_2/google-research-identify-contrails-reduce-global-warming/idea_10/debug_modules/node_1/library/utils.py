import os
import random
import numpy as np
import torch
from library.config import Config


def set_seed(seed=Config.SEED):
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

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def rle_encode(mask):
    """
    Encodes a binary mask using Run-Length Encoding (RLE) formatted for submission.

    The pixels are numbered from top to bottom, then left to right (column-major order).

    Args:
        mask (np.ndarray): Binary mask of shape (H, W) or (H, W, 1).
                           Values should be 0 (background) or 1 (contrail).

    Returns:
        str: Space-delimited list of pairs (start_pixel, length) or '-' if the mask is empty.
    """
    # Ensure mask is 2D
    if mask.ndim == 3:
        mask = mask.squeeze(-1)

    # Flatten in column-major order (Fortran-style) as per competition spec
    pixels = mask.flatten(order="F")

    # If no pixels are masked, return '-'
    if np.sum(pixels) == 0:
        return "-"

    # Pad with 0s at start and end to detect transitions at edges
    # We use int8 to save memory during concatenation if mask is bool/int
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where values change
    # pixels[1:] != pixels[:-1] gives boolean array of transitions
    # np.where returns indices in the padded array
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # runs[0::2] are start indices (inclusive)
    # runs[1::2] are end indices (exclusive) in the 1-based indexing logic

    # Calculate lengths: end - start
    runs[1::2] -= runs[0::2]

    # Format as space-separated string
    return " ".join(str(x) for x in runs)


def dice_coefficient(y_pred, y_true, smooth=1e-6):
    """
    Computes the Dice coefficient between predictions and ground truth.
    Useful for batch-level monitoring or soft-dice loss components.

    Args:
        y_pred (torch.Tensor): Predicted probabilities or binary mask.
        y_true (torch.Tensor): Ground truth mask.
        smooth (float): Smoothing factor to prevent division by zero.

    Returns:
        float: The Dice coefficient.
    """
    # Flatten tensors
    y_pred_f = y_pred.view(-1)
    y_true_f = y_true.view(-1)

    intersection = (y_pred_f * y_true_f).sum()
    union = y_pred_f.sum() + y_true_f.sum()

    return (2.0 * intersection + smooth) / (union + smooth)


class GlobalDiceTracker:
    """
    Accumulates statistics to compute the Global Dice Coefficient over an entire dataset.

    Formula: 2 * |X n Y| / (|X| + |Y|)
    where X is the set of all predicted pixels and Y is the set of all ground truth pixels
    across all images in the set.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.intersection = 0.0
        self.union = 0.0

    def update(self, y_pred, y_true):
        """
        Updates the tracker with a batch of predictions and targets.

        Args:
            y_pred (torch.Tensor or np.ndarray): Predicted binary masks.
            y_true (torch.Tensor or np.ndarray): Ground truth binary masks.
        """
        # Convert to numpy if tensors
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()

        # Ensure binary (assuming thresholding has been applied if needed,
        # or inputs are already 0/1)
        y_pred = (y_pred > 0.5).astype(np.float32)
        y_true = (y_true > 0.5).astype(np.float32)

        # Accumulate intersection and areas
        self.intersection += np.sum(y_pred * y_true)
        self.union += np.sum(y_pred) + np.sum(y_true)

    def compute(self):
        """
        Computes the current Global Dice score.

        Returns:
            float: Global Dice coefficient.
        """
        if self.union == 0:
            return 0.0
        return 2.0 * self.intersection / self.union


class AverageMeter:
    """
    Computes and stores the average and current value of a metric.
    Useful for tracking loss during training loops.
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
