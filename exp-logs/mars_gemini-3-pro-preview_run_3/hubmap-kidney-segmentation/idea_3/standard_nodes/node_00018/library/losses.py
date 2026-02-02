import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepSupervisionLoss(nn.Module):
    """
    Combined BCE + Dice Loss with support for Deep Supervision.

    This loss function computes a weighted sum of losses from multiple model outputs
    (corresponding to different decoder levels in U-Net++). Each individual loss
    is a combination of Binary Cross Entropy (BCE) and a differentiable Dice Loss.
    """

    def __init__(
        self, weights=[1.0, 0.5, 0.25], bce_weight=0.5, dice_weight=0.5, smooth=1e-5
    ):
        """
        Args:
            weights (list): Weights for the deep supervision outputs. The first weight
                            corresponds to the final output, subsequent weights to
                            intermediate outputs.
            bce_weight (float): Weight for the BCE component of the loss.
            dice_weight (float): Weight for the Dice component of the loss.
            smooth (float): Smoothing factor for Dice loss to avoid division by zero.
        """
        super(DeepSupervisionLoss, self).__init__()
        self.weights = weights
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, preds, targets):
        """
        Computes the weighted loss.

        Args:
            preds (torch.Tensor or list of torch.Tensor): Model predictions.
                If list, assumed to be [final_output, aux_output1, aux_output2, ...].
                Predictions are logits (before sigmoid).
            targets (torch.Tensor): Ground truth masks. Shape (B, H, W) or (B, 1, H, W).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Ensure targets have the channel dimension: (B, H, W) -> (B, 1, H, W)
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)

        # Cast targets to float for BCE and calculation
        targets = targets.float()

        # Handle Deep Supervision (list of outputs)
        if isinstance(preds, list):
            loss = 0.0
            # Iterate over predictions and corresponding weights
            for i, pred in enumerate(preds):
                # Use 0.0 weight if we run out of specified weights
                w = self.weights[i] if i < len(self.weights) else 0.0

                if w > 0.0:
                    loss += w * self._compute_single_loss(pred, targets)
            return loss
        else:
            # Single output case
            return self._compute_single_loss(preds, targets)

    def _compute_single_loss(self, pred, target):
        """
        Computes the combined BCE + Dice loss for a single prediction tensor.

        Args:
            pred (torch.Tensor): Logits from the model (B, 1, H, W).
            target (torch.Tensor): Ground truth mask (B, 1, H, W).

        Returns:
            torch.Tensor: Combined loss.
        """
        # 1. Binary Cross Entropy Loss
        bce_loss = self.bce(pred, target)

        # 2. Differentiable Dice Loss
        pred_sigmoid = torch.sigmoid(pred)

        # Flatten tensors for Dice calculation: (B, C, H, W) -> (B, -1)
        # We compute Dice per sample in the batch and average
        pred_flat = pred_sigmoid.view(pred_sigmoid.size(0), -1)
        target_flat = target.view(target.size(0), -1)

        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)

        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice_score.mean()

        # Combined Loss
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss
