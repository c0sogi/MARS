import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEDiceLoss(nn.Module):
    """
    A custom loss function that combines Binary Cross Entropy (BCE) with Logits Loss
    and Dice Loss. This is effective for binary segmentation tasks, particularly
    when dealing with class imbalance.
    """

    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1.0):
        """
        Args:
            bce_weight (float): Weight assigned to the BCE loss component.
            dice_weight (float): Weight assigned to the Dice loss component.
            smooth (float): Smoothing factor for Dice calculation to avoid division by zero.
        """
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.bce_func = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        """
        Calculates the combined loss.

        Args:
            logits (torch.Tensor): Raw model predictions (before sigmoid) of shape (N, C, H, W) or (N, H, W).
            targets (torch.Tensor): Ground truth binary masks of shape (N, C, H, W) or (N, H, W).

        Returns:
            torch.Tensor: The calculated scalar loss.
        """
        # Ensure targets are float for BCE calculation
        targets = targets.float()

        # Align dimensions: if logits are (N, 1, H, W) and targets are (N, H, W), unsqueeze targets
        if logits.dim() == 4 and targets.dim() == 3:
            targets = targets.unsqueeze(1)

        # --- BCE Loss ---
        # BCEWithLogitsLoss handles the sigmoid internally for numerical stability
        bce_loss = self.bce_func(logits, targets)

        # --- Dice Loss ---
        # Apply sigmoid to get probabilities for Dice calculation
        probs = torch.sigmoid(logits)

        # Flatten the tensors to (N, -1) to calculate Dice per sample in the batch
        # This treats the spatial dimensions and channels as a single vector per image
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)

        intersection = (probs_flat * targets_flat).sum(dim=1)
        union = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)

        # Calculate Dice score
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Dice loss is 1 - Dice score. We take the mean across the batch.
        dice_loss = 1.0 - dice_score.mean()

        # --- Combined Loss ---
        total_loss = (self.bce_weight * bce_loss) + (self.dice_weight * dice_loss)

        return total_loss
