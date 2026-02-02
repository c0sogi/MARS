import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEDiceLoss(nn.Module):
    """
    Combined Binary Cross Entropy and Dice Loss for Semantic Segmentation.
    Supports Deep Supervision by accepting a list of model outputs.
    """

    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1.0, ds_weights=None):
        """
        Args:
            bce_weight (float): Weight for the BCE component.
            dice_weight (float): Weight for the Dice component.
            smooth (float): Smoothing factor for Dice calculation to avoid division by zero.
            ds_weights (list of float, optional): Weights for Deep Supervision heads.
                                                  If None, heads are averaged equally.
        """
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.ds_weights = ds_weights
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, outputs, targets):
        """
        Args:
            outputs: Model output. Can be a single tensor (N, C, H, W) or a list/tuple of tensors
                     for deep supervision.
            targets: Ground truth mask. Shape (N, H, W) or (N, C, H, W).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Ensure targets are float and have the correct channel dimension
        if targets.dtype != torch.float32:
            targets = targets.float()

        # If targets is (N, H, W), unsqueeze to (N, 1, H, W) to match logits
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)

        # Handle Deep Supervision (list of outputs)
        if isinstance(outputs, (list, tuple)):
            total_loss = 0
            num_heads = len(outputs)

            # Determine weights for each head
            if self.ds_weights is None:
                # Default: Equal weighting (average)
                weights = [1.0 / num_heads] * num_heads
            else:
                if len(self.ds_weights) != num_heads:
                    raise ValueError(
                        f"Length of ds_weights ({len(self.ds_weights)}) "
                        f"must match number of outputs ({num_heads})."
                    )
                weights = self.ds_weights

            for output, weight in zip(outputs, weights):
                total_loss += weight * self._compute_single_loss(output, targets)

            return total_loss

        # Handle Single Output
        else:
            return self._compute_single_loss(outputs, targets)

    def _compute_single_loss(self, logits, targets):
        """
        Computes the combined BCE and Dice loss for a single output tensor.
        """
        # 1. Binary Cross Entropy Loss
        bce = self.bce_loss(logits, targets)

        # 2. Dice Loss
        # Apply sigmoid to convert logits to probabilities
        probs = torch.sigmoid(logits)

        # Flatten tensors for global Dice calculation (batch-wise)
        # This is often more stable than per-image Dice for loss optimization
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        union = probs_flat.sum() + targets_flat.sum()

        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice_score

        # Weighted combination
        return (self.bce_weight * bce) + (self.dice_weight * dice_loss)
