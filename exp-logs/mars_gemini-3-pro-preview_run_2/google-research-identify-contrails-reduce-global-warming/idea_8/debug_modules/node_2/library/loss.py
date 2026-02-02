import torch
import torch.nn as nn
from library.utils import dice_coeff


class BatchDiceLoss(nn.Module):
    """
    Computes the Dice Loss over the entire flattened batch.
    Treats the batch as a single volume to stabilize gradients.

    Formula: Loss = 1 - Dice_Coefficient
    """

    def __init__(self, smooth=1e-6):
        super(BatchDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): Raw model outputs (B, C, H, W).
            targets (torch.Tensor): Binary ground truth masks (B, C, H, W).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Apply sigmoid to convert logits to probabilities
        probs = torch.sigmoid(logits)

        # Calculate global dice score.
        # dice_coeff flattens the input tensors, effectively computing batch-level Dice.
        dice_score = dice_coeff(probs, targets, smooth=self.smooth)

        return 1.0 - dice_score


class HybridLoss(nn.Module):
    """
    Hybrid Loss function combining Binary Cross Entropy and Batch-level Dice Loss.

    Formula: L_total = L_BCE + L_BatchDice
    """

    def __init__(self, smooth=1e-6):
        super(HybridLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.batch_dice = BatchDiceLoss(smooth=smooth)

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): Raw model outputs (B, C, H, W).
            targets (torch.Tensor): Binary ground truth masks (B, C, H, W).

        Returns:
            torch.Tensor: Combined scalar loss.
        """
        bce_loss = self.bce(logits, targets)
        dice_loss = self.batch_dice(logits, targets)

        return bce_loss + dice_loss
