import torch
import torch.nn as nn
import numpy as np
from library.config import Config


class BCEDiceLoss(nn.Module):
    """
    Combined Binary Cross Entropy and Dice Loss.
    Useful for segmentation tasks with class imbalance.
    """

    def __init__(self, smooth=1e-6):
        """
        Args:
            smooth (float): Smoothing factor to prevent division by zero in Dice calculation.
        """
        super(BCEDiceLoss, self).__init__()
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Model predictions (logits) of shape (B, C, H, W).
            targets (torch.Tensor): Ground truth masks of shape (B, C, H, W).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Binary Cross Entropy Loss
        bce_loss = self.bce(inputs, targets)

        # Dice Loss
        # Apply sigmoid to convert logits to probabilities
        inputs_prob = torch.sigmoid(inputs)

        # Flatten the tensors to compute global Dice (or per-batch Dice)
        inputs_flat = inputs_prob.view(-1)
        targets_flat = targets.view(-1)

        intersection = (inputs_flat * targets_flat).sum()
        dice_score = (2.0 * intersection + self.smooth) / (
            inputs_flat.sum() + targets_flat.sum() + self.smooth
        )

        dice_loss = 1.0 - dice_score

        return bce_loss + dice_loss


def fbeta_score_numpy(preds, targets, beta=0.5, threshold=0.5, epsilon=1e-6):
    """
    Calculates the F-beta score for binary segmentation using NumPy.
    The F0.5 score weights precision higher than recall.

    Args:
        preds (np.ndarray): Model predictions (probabilities or logits).
        targets (np.ndarray): Ground truth binary masks.
        beta (float): The beta value for the F-score (default 0.5).
        threshold (float): Threshold to convert probabilities to binary mask.
        epsilon (float): Small constant to avoid division by zero.

    Returns:
        float: The F-beta score.
    """
    # Binarize predictions
    preds_bin = (preds > threshold).astype(np.uint8)
    targets_bin = (targets > 0.5).astype(np.uint8)

    # Flatten arrays
    preds_flat = preds_bin.flatten()
    targets_flat = targets_bin.flatten()

    # Calculate True Positives, False Positives, False Negatives
    tp = np.sum(preds_flat * targets_flat)
    fp = np.sum(preds_flat * (1 - targets_flat))
    fn = np.sum((1 - preds_flat) * targets_flat)

    # F-beta formula: ((1 + beta^2) * TP) / ((1 + beta^2) * TP + beta^2 * FN + FP)
    beta_sq = beta**2
    numerator = (1 + beta_sq) * tp
    denominator = (1 + beta_sq) * tp + (beta_sq * fn) + fp

    score = numerator / (denominator + epsilon)

    return float(score)
