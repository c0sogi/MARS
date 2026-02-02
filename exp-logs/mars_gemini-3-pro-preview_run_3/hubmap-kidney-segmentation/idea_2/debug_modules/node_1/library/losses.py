import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Differentiable Dice Loss for binary segmentation.
    Expects logits as input.
    """

    def __init__(self, smooth=1e-7):
        """
        Args:
            smooth (float): Smoothing factor to prevent division by zero.
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): Model output logits of shape (B, 1, H, W).
            targets (torch.Tensor): Ground truth binary mask of shape (B, 1, H, W).

        Returns:
            torch.Tensor: Scalar Dice loss.
        """
        # Apply sigmoid to convert logits to probabilities
        probs = torch.sigmoid(logits)

        # Flatten the tensors
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()

        dice_score = (2.0 * intersection + self.smooth) / (
            probs_flat.sum() + targets_flat.sum() + self.smooth
        )

        return 1.0 - dice_score


class DiceBCELoss(nn.Module):
    """
    Combination of Binary Cross Entropy (BCE) and Dice Loss.
    """

    def __init__(self, weight_bce=0.5, weight_dice=0.5, smooth=1e-7):
        """
        Args:
            weight_bce (float): Weight for the BCE component.
            weight_dice (float): Weight for the Dice component.
            smooth (float): Smoothing factor for Dice loss.
        """
        super(DiceBCELoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth)
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): Model output logits.
            targets (torch.Tensor): Ground truth masks.

        Returns:
            torch.Tensor: Weighted sum of BCE and Dice loss.
        """
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)

        return self.weight_bce * bce_loss + self.weight_dice * dice_loss


class DeepSupervisionLoss(nn.Module):
    """
    Loss function for Deep Supervision.
    Computes the weighted average of DiceBCELoss for a list of outputs.
    """

    def __init__(self, weights=None):
        """
        Args:
            weights (list of float, optional): Weights for each output head.
                                              If None, defaults to equal weighting.
        """
        super(DeepSupervisionLoss, self).__init__()
        self.criterion = DiceBCELoss()
        self.weights = weights

    def forward(self, preds, targets):
        """
        Args:
            preds (list of torch.Tensor or torch.Tensor): List of model outputs or single output.
            targets (torch.Tensor): Ground truth mask.

        Returns:
            torch.Tensor: Aggregated loss.
        """
        # Handle case where preds is a single tensor (e.g., validation/inference)
        if not isinstance(preds, list):
            return self.criterion(preds, targets)

        loss = 0.0

        # Determine weights
        if self.weights is None or len(self.weights) != len(preds):
            # Default to equal weights if not provided or mismatch
            current_weights = [1.0] * len(preds)
        else:
            current_weights = self.weights

        total_weight = sum(current_weights)

        # Compute weighted loss for each head
        for i, pred in enumerate(preds):
            w = current_weights[i]
            # Ensure target is on same device (it should be)
            term_loss = self.criterion(pred, targets)
            loss += w * term_loss

        # Normalize by sum of weights
        return loss / total_weight
