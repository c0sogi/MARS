import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """
    Custom Dice Loss implementation for binary segmentation.
    Calculates the Dice coefficient (F1 score) based loss.
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
            logits (torch.Tensor): Model output logits of shape (N, C, H, W) or (N, 1, H, W).
            targets (torch.Tensor): Ground truth masks of shape (N, C, H, W) or (N, 1, H, W).

        Returns:
            torch.Tensor: Scalar loss value (1 - Dice).
        """
        # Apply sigmoid to convert logits to probabilities
        probs = torch.sigmoid(logits)

        # Flatten the tensors to compute the global dice score for the batch
        # This treats the batch as a single volume, which is common for segmentation stability
        # and helps when some individual patches in the batch might be empty.
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        union = probs_flat.sum() + targets_flat.sum()

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        return 1.0 - dice


class BCEDiceLoss(nn.Module):
    """
    Composite loss function combining Binary Cross Entropy (BCE) and Dice Loss.
    Used to optimize for both pixel-wise accuracy and segmentation overlap,
    helping to address class imbalance.
    """

    def __init__(self, bce_weight=0.5, smooth=1e-6):
        """
        Args:
            bce_weight (float): Weight assigned to the BCE component.
                                The Dice component will receive (1 - bce_weight).
            smooth (float): Smoothing factor for the Dice Loss.
        """
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(smooth=smooth)

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): Model output logits.
            targets (torch.Tensor): Ground truth binary masks.

        Returns:
            torch.Tensor: Weighted composite loss.
        """
        # Ensure targets are float for BCEWithLogitsLoss
        targets = targets.float()

        loss_bce = self.bce_loss(logits, targets)
        loss_dice = self.dice_loss(logits, targets)

        return self.bce_weight * loss_bce + (1.0 - self.bce_weight) * loss_dice
