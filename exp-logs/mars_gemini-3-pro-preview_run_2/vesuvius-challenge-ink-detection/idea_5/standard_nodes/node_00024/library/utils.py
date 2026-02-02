import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def rle_encoding(mask):
    """
    Converts a binary mask to Run-Length Encoding (RLE).
    The pixels are numbered from left to right, then top to bottom.

    Args:
        mask (numpy.ndarray): Binary mask (0 or 1).

    Returns:
        str: Space-delimited list of pairs (start_position, run_length).
    """
    # Flatten in row-major order (C-style) as per task description
    pixels = mask.flatten()

    # Prepend and append 0 to detect transitions at start/end
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where value changes
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # Calculate lengths: runs[1::2] are ends, runs[::2] are starts
    runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)


def calculate_fbeta(pred, target, beta=0.5, threshold=0.5, smooth=1e-6):
    """
    Calculates the F-beta score.

    Args:
        pred (torch.Tensor): Predicted probabilities (0-1).
        target (torch.Tensor): Ground truth binary mask.
        beta (float): Beta value for F-score (default 0.5).
        threshold (float): Threshold to binarize predictions (default 0.5).
        smooth (float): Smoothing factor to avoid division by zero.

    Returns:
        float: F-beta score.
    """
    # Ensure inputs are flattened and on the same device
    pred = pred.view(-1)
    target = target.view(-1)

    # Binarize predictions based on threshold
    pred_bin = (pred > threshold).float()

    # Calculate True Positives, False Positives, False Negatives
    tp = (pred_bin * target).sum()
    fp = (pred_bin * (1 - target)).sum()
    fn = ((1 - pred_bin) * target).sum()

    beta_sq = beta**2

    # F-beta formula
    numerator = (1 + beta_sq) * tp
    denominator = (1 + beta_sq) * tp + beta_sq * fn + fp

    score = numerator / (denominator + smooth)

    return score.item()


class DiceBCELoss(nn.Module):
    """
    Combined Dice and Binary Cross Entropy Loss.
    Expects logits as input for numerical stability.
    """

    def __init__(self, weight=None, size_average=True, smooth=1e-6):
        super(DiceBCELoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Model predictions (logits).
            targets (torch.Tensor): Ground truth binary masks.

        Returns:
            torch.Tensor: Combined loss value.
        """
        # Flatten inputs and targets
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # Binary Cross Entropy (with Logits for stability)
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="mean")

        # Dice Loss
        # Apply sigmoid to convert logits to probabilities for Dice calculation
        inputs_sigmoid = torch.sigmoid(inputs)
        intersection = (inputs_sigmoid * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (
            inputs_sigmoid.sum() + targets.sum() + self.smooth
        )

        # Combined Loss
        return bce_loss + (1 - dice)
