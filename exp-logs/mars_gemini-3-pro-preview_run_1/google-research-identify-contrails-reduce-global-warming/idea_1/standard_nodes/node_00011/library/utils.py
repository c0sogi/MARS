import numpy as np
import torch
from library.config import Config


def rle_encode(mask):
    """
    Encodes a binary mask using Run-Length Encoding (RLE).
    The pixels are numbered from top to bottom, then left to right (Fortran order).

    Args:
        mask (np.ndarray or torch.Tensor): Binary mask of shape (H, W).
                                           1 - mask, 0 - background.

    Returns:
        str: Space delimited list of pairs (start length) or '-' if empty.
    """
    # Convert tensor to numpy if necessary
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    # Ensure mask is binary integer type
    mask = mask.astype(np.int8)

    # Flatten in column-major order (Fortran-style) as per task description
    pixels = mask.flatten(order="F")

    # Check if empty
    if np.sum(pixels) == 0:
        return "-"

    # Concatenate [0] at start and end to detect runs at boundaries
    # This simplifies logic: every transition from 0 to 1 is a start, 1 to 0 is an end
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where values change
    # np.where returns indices in the padded array
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # Calculate lengths
    # runs[::2] are start indices (1-based because of +1 and 0-padding logic alignment)
    # runs[1::2] are end indices (exclusive in 0-based, but effectively end+1)
    # The length is simply end_index - start_index
    runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)


def dice_coefficient(y_pred, y_true, threshold=None, smooth=1e-6):
    """
    Calculates the Dice Coefficient for a single batch or image.
    Useful for loss calculation or batch-wise logging.

    Args:
        y_pred (torch.Tensor): Predicted probabilities or binary mask.
        y_true (torch.Tensor): Ground truth binary mask.
        threshold (float, optional): Threshold to binarize predictions.
                                     If None, calculates soft dice (for probabilities).
        smooth (float): Smoothing factor to prevent division by zero.

    Returns:
        float: Dice coefficient.
    """
    # Apply threshold if provided (for binary metric)
    if threshold is not None:
        y_pred = (y_pred > threshold).float()

    # Flatten tensors to ensure correct computation regardless of shape
    y_pred_f = y_pred.view(-1)
    y_true_f = y_true.view(-1)

    intersection = (y_pred_f * y_true_f).sum()
    union = y_pred_f.sum() + y_true_f.sum()

    return (2.0 * intersection + smooth) / (union + smooth)


class GlobalDiceMetric:
    """
    Accumulates statistics to compute the Global Dice Coefficient
    over an entire dataset (e.g., validation epoch).

    Metric Formula: 2 * |X n Y| / (|X| + |Y|)
    where X is the set of predicted pixels in the entire dataset,
    and Y is the set of ground truth pixels in the entire dataset.
    """

    def __init__(self):
        self.intersection = 0.0
        self.union = 0.0

    def update(self, y_pred, y_true, threshold=0.5):
        """
        Update internal stats with a batch of predictions.

        Args:
            y_pred (torch.Tensor): Predicted probabilities.
            y_true (torch.Tensor): Ground truth masks.
            threshold (float): Threshold to binarize predictions.
        """
        # Ensure inputs are on the same device
        if y_pred.device != y_true.device:
            y_true = y_true.to(y_pred.device)

        # Binarize predictions
        preds = (y_pred > threshold).float()
        targets = y_true.float()

        # Compute intersection and union sums for this batch
        inter = (preds * targets).sum().item()
        pred_sum = preds.sum().item()
        true_sum = targets.sum().item()

        self.intersection += inter
        self.union += pred_sum + true_sum

    def compute(self):
        """Compute the final global dice score."""
        if self.union == 0:
            return 0.0
        return (2.0 * self.intersection) / self.union

    def reset(self):
        self.intersection = 0.0
        self.union = 0.0


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
