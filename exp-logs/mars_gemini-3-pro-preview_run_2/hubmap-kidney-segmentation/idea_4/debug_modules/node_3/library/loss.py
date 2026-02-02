import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DiceLoss(nn.Module):
    """
    Soft Dice Loss for binary segmentation.
    Expects logits as input.
    """

    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Args:
            logits: (torch.Tensor) Model output logits of shape (N, H, W) or (N, 1, H, W).
            targets: (torch.Tensor) Ground truth binary mask of same shape.
        """
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits)

        # Flatten tensors
        probs = probs.reshape(-1)
        targets = targets.reshape(-1)

        intersection = (probs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (
            probs.sum() + targets.sum() + self.smooth
        )

        return 1.0 - dice


class HybridBCEDiceLoss(nn.Module):
    """
    Hybrid loss combining Binary Cross Entropy and Dice Loss.
    Used for the primary glomerulus segmentation task to handle class imbalance.
    """

    def __init__(self):
        super(HybridBCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits, targets):
        """
        Args:
            logits: (torch.Tensor) Model output logits.
            targets: (torch.Tensor) Ground truth binary mask.
        """
        # Ensure targets are float for BCE and Dice calculations
        targets = targets.float()

        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)

        # Summing losses (equal weighting implicitly, or could be parameterized)
        return bce_loss + dice_loss


class MultiTaskLoss(nn.Module):
    """
    Multi-Task Loss function aggregating Primary (Glomerulus) and Auxiliary (Cortex) losses.

    Structure:
        Total Loss = L_primary + lambda * L_auxiliary

    Where:
        L_primary = HybridBCEDiceLoss (on Channel 0)
        L_auxiliary = BCEWithLogitsLoss (on Channel 1)
        lambda = Config.AUX_LOSS_WEIGHT
    """

    def __init__(self):
        super(MultiTaskLoss, self).__init__()
        self.primary_loss_fn = HybridBCEDiceLoss()
        self.aux_loss_fn = nn.BCEWithLogitsLoss()
        self.aux_weight = Config.AUX_LOSS_WEIGHT

    def forward(self, outputs, targets):
        """
        Args:
            outputs: (torch.Tensor) Model predictions of shape (B, 2, H, W).
                     Channel 0: Glomerulus logits.
                     Channel 1: Anatomical Structure (Cortex) logits.
            targets: (torch.Tensor) Ground truth masks of shape (B, 2, H, W).
                     Channel 0: Glomerulus mask.
                     Channel 1: Anatomical Structure (Cortex) mask.
        """
        # Validate shapes
        assert (
            outputs.shape[1] == 2
        ), f"Expected 2 output channels, got {outputs.shape[1]}"
        assert (
            targets.shape[1] == 2
        ), f"Expected 2 target channels, got {targets.shape[1]}"

        # Split channels
        # Channel 0: Primary Task (Glomerulus)
        primary_logits = outputs[:, 0, :, :]
        primary_targets = targets[:, 0, :, :]

        # Channel 1: Auxiliary Task (Cortex)
        aux_logits = outputs[:, 1, :, :]
        aux_targets = targets[:, 1, :, :]

        # Calculate individual losses
        # Ensure targets are float
        loss_primary = self.primary_loss_fn(primary_logits, primary_targets)
        loss_aux = self.aux_loss_fn(aux_logits, aux_targets.float())

        # Aggregate
        total_loss = loss_primary + (self.aux_weight * loss_aux)

        return total_loss
