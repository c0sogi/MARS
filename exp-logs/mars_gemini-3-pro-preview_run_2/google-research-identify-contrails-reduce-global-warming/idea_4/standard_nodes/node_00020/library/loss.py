import torch
import torch.nn as nn
from library.config import Config


class GlobalBatchDiceLoss(nn.Module):
    """
    Computes the Global Dice Coefficient Loss for a batch of predictions.

    This loss treats the entire batch as a single volume, flattening all samples
    into one large vector. This stabilizes the gradient and metric calculation,
    especially when many samples in the batch are empty (common in contrail detection).

    Formula:
        Loss = 1 - (2 * Intersection + epsilon) / (Cardinality + epsilon)
        where Intersection and Cardinality are summed over the entire batch.
    """

    def __init__(self, smooth=1e-6):
        super(GlobalBatchDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Predicted probabilities (0-1). Shape (B, C, H, W) or (B, H, W).
            targets (torch.Tensor): Ground truth binary masks (0 or 1). Shape (B, C, H, W) or (B, H, W).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Flatten inputs and targets to (N,)
        # This aggregates all pixels from all images in the batch
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (inputs_flat * targets_flat).sum()
        # Cardinality: Sum of probabilities + Sum of true pixels
        cardinality = inputs_flat.sum() + targets_flat.sum()

        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

        return 1.0 - dice_score


class BCEDiceLoss(nn.Module):
    """
    Hybrid Loss: BCEWithLogitsLoss + GlobalBatchDiceLoss.

    Combines pixel-wise stability of BCE with global metric optimization of Dice.
    """

    def __init__(self, smooth=1e-6):
        super(BCEDiceLoss, self).__init__()
        self.bce_fn = nn.BCEWithLogitsLoss()
        self.dice_fn = GlobalBatchDiceLoss(smooth=smooth)
        self.last_metrics = {}

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): Model output logits.
            targets (torch.Tensor): Ground truth masks.
        """
        # BCE Loss (Pixel-wise)
        bce = self.bce_fn(logits, targets)

        # Dice Loss (Global Batch)
        # Apply sigmoid to logits for Dice calculation
        probs = torch.sigmoid(logits)
        dice = self.dice_fn(probs, targets)

        total_loss = bce + dice

        self.last_metrics = {
            "loss_total": total_loss.item(),
            "loss_bce": bce.item(),
            "loss_dice": dice.item(),
        }

        return total_loss


class SoftGatedLoss(nn.Module):
    """
    Loss function for Soft-Gated Multi-Task ResNet U-Net.
    Handles dictionary output and probability inputs (since model has Sigmoid).
    """

    def __init__(self, smooth=1e-6):
        super(SoftGatedLoss, self).__init__()
        # Use BCELoss because model outputs probabilities (Sigmoid applied)
        self.bce_fn = nn.BCELoss()
        self.dice_fn = GlobalBatchDiceLoss(smooth=smooth)
        self.last_metrics = {}

    def forward(self, outputs, targets):
        """
        Args:
            outputs (dict or Tensor): Model output. If dict, expects 'mask' key.
                                      Values should be probabilities [0, 1].
            targets (Tensor): Ground truth masks.
        """
        if isinstance(outputs, dict):
            preds = outputs["mask"]
        else:
            preds = outputs

        # Clamp predictions to avoid log(0) in BCELoss
        preds = torch.clamp(preds, min=1e-7, max=1.0 - 1e-7)

        # BCE Loss
        bce = self.bce_fn(preds, targets)

        # Dice Loss
        dice = self.dice_fn(preds, targets)

        total_loss = bce + dice

        self.last_metrics = {
            "loss_total": total_loss.item(),
            "loss_bce": bce.item(),
            "loss_dice": dice.item(),
        }

        return total_loss
