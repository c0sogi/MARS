import torch
import torch.nn as nn
from library.config import Config


class BCEDiceLoss(nn.Module):
    """
    Combined Binary Cross Entropy and Soft Dice Loss.
    Used for segmentation tasks to optimize both pixel-wise accuracy and geometric overlap.
    """

    def __init__(self, bce_weight=None, dice_weight=None, smooth=1e-6):
        """
        Initialize the loss function.

        Args:
            bce_weight (float, optional): Weight for the BCE component.
                                          Defaults to Config.BCE_WEIGHT.
            dice_weight (float, optional): Weight for the Dice component.
                                           Defaults to Config.DICE_WEIGHT.
            smooth (float): Smoothing factor for Dice calculation to prevent division by zero.
        """
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight if bce_weight is not None else Config.BCE_WEIGHT
        self.dice_weight = (
            dice_weight if dice_weight is not None else Config.DICE_WEIGHT
        )
        self.smooth = smooth

        # Use BCEWithLogitsLoss for numerical stability in mixed precision
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, y_pred, y_true):
        """
        Calculate the combined loss.

        Args:
            y_pred (torch.Tensor): Predicted LOGITS of shape (B, C, H, W).
            y_true (torch.Tensor): Ground truth masks of shape (B, C, H, W).

        Returns:
            torch.Tensor: Weighted sum of BCE and Dice loss.
        """
        # Ensure target is float for BCELoss
        y_true = y_true.type_as(y_pred)

        # 1. Binary Cross Entropy Loss (takes logits)
        bce = self.bce_loss(y_pred, y_true)

        # 2. Soft Dice Loss
        # Apply Sigmoid to logits to get probabilities for Dice calculation
        y_pred_prob = torch.sigmoid(y_pred)

        batch_size = y_pred_prob.size(0)
        num_classes = y_pred_prob.size(1)

        y_pred_flat = y_pred_prob.view(batch_size, num_classes, -1)
        y_true_flat = y_true.view(batch_size, num_classes, -1)

        intersection = (y_pred_flat * y_true_flat).sum(dim=2)
        union = y_pred_flat.sum(dim=2) + y_true_flat.sum(dim=2)

        # Dice coefficient per channel per sample
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Dice Loss = 1 - Dice Score
        # We average the dice loss over the batch and classes
        dice_loss = 1.0 - dice_score.mean()

        # 3. Combined Loss
        total_loss = (self.bce_weight * bce) + (self.dice_weight * dice_loss)

        return total_loss
