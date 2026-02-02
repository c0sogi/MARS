import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation tasks.
    Calculates 1 - Dice Coefficient between logits and binary targets.
    Expects logits as input (applies sigmoid internally).
    """

    def __init__(self, smooth=1e-6):
        """
        Args:
            smooth (float): Smoothing factor to prevent division by zero.
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): Raw output from the model (before sigmoid), shape (N, C, H, W) or (N, H, W).
            targets (torch.Tensor): Ground truth binary masks, same shape as logits.

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Apply sigmoid to convert logits to probabilities
        probs = torch.sigmoid(logits)

        # Flatten the tensors to compute the global dice over the batch
        # This aligns with the "Global Dice" metric definition where X and Y are sets of pixels
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        union = probs_flat.sum() + targets_flat.sum()

        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)

        return 1.0 - dice_score


class BCEDiceLoss(nn.Module):
    """
    Combined Binary Cross Entropy and Dice Loss.
    Useful for stabilizing training in segmentation tasks with class imbalance.
    """

    def __init__(self, bce_weight=1.0, dice_weight=1.0, smooth=1e-6):
        """
        Args:
            bce_weight (float): Weight for the BCE component.
            dice_weight (float): Weight for the Dice component.
            smooth (float): Smoothing factor for Dice Loss.
        """
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(smooth=smooth)

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): Raw output from the model.
            targets (torch.Tensor): Ground truth binary masks.

        Returns:
            torch.Tensor: Combined scalar loss.
        """
        # Ensure targets are float for BCE
        if targets.dtype != logits.dtype:
            targets = targets.type_as(logits)

        loss_bce = self.bce_loss(logits, targets)
        loss_dice = self.dice_loss(logits, targets)

        return (self.bce_weight * loss_bce) + (self.dice_weight * loss_dice)


def dice_coefficient(logits, targets, threshold=0.5, smooth=1e-6):
    """
    Calculates the Dice Coefficient metric for evaluation.
    Converts logits to binary predictions using a threshold.

    Formula: 2 * |X n Y| / (|X| + |Y|)

    Args:
        logits (torch.Tensor): Raw output from the model.
        targets (torch.Tensor): Ground truth binary masks.
        threshold (float): Threshold to convert probabilities to binary mask.
        smooth (float): Smoothing factor.

    Returns:
        float: The Dice coefficient.
    """
    with torch.no_grad():
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()

        # Flatten to treat as a single set of pixels (Global Dice over the batch)
        preds_flat = preds.view(-1)
        targets_flat = targets.view(-1)

        intersection = (preds_flat * targets_flat).sum()
        union = preds_flat.sum() + targets_flat.sum()

        dice = (2.0 * intersection + smooth) / (union + smooth)

        return dice.item()
