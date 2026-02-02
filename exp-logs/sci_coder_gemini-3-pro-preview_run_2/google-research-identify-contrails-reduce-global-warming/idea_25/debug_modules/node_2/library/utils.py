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
        # Ensure deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def rle_encode(mask):
    """
    Encodes a binary mask using Run-Length Encoding (RLE).
    The pixels are numbered from top to bottom, then left to right (Column-Major).

    Args:
        mask (np.ndarray or torch.Tensor): Binary mask of shape (H, W).

    Returns:
        str: Space-delimited list of pairs 'start length' or '-' if empty.
    """
    # Convert Tensor to numpy if needed
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    # Ensure mask is binary (0 or 1)
    mask = (mask > 0).astype(np.uint8)

    # Flatten column-major (Fortran style) as per task description
    pixels = mask.flatten(order="F")

    # If mask is empty
    if np.sum(pixels) == 0:
        return "-"

    # Add padding to detect changes at the beginning and end
    # We pad with 0 at both ends
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where the value changes (0->1 or 1->0)
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # runs[0::2] are the starts of 1s
    # runs[1::2] are the ends of 1s (exclusive in 0-based, or start of next 0)

    # Calculate lengths: end - start
    runs[1::2] -= runs[0::2]

    # Format as space-delimited string
    return " ".join(str(x) for x in runs)


class MetricMonitor:
    """
    A utility class to track running averages of metrics (like Loss)
    during training/validation loops.
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

    def __str__(self):
        # Print full precision as requested
        return str(self.avg)


class GlobalDiceTracker:
    """
    Tracks the Global Dice Coefficient over an entire dataset/epoch.
    Formula: 2 * |X n Y| / (|X| + |Y|)
    where X is the set of all predicted pixels and Y is the set of all ground truth pixels.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.intersection = 0.0
        self.union = 0.0

    def update(self, y_pred, y_true, threshold=0.5):
        """
        Updates the intersection and union counts for a batch.

        Args:
            y_pred (torch.Tensor): Predicted probabilities or logits.
            y_true (torch.Tensor): Ground truth binary masks.
            threshold (float): Threshold to binarize predictions.
        """
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu()
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu()

        # Binarize predictions
        pred_mask = (y_pred > threshold).float()
        true_mask = y_true.float()

        # Update sums
        self.intersection += (pred_mask * true_mask).sum().item()
        self.union += pred_mask.sum().item() + true_mask.sum().item()

    def compute(self):
        """
        Computes the global Dice score based on accumulated stats.
        """
        if self.union == 0:
            # If both prediction and ground truth are empty everywhere, score is 1.0
            return 1.0

        return 2.0 * self.intersection / self.union
