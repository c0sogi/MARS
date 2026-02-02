import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Implements the Dice Loss for binary segmentation.
    Loss = 1 - Dice_Coefficient
    Dice_Coefficient = (2 * Intersection + Smooth) / (Union + Smooth)
    """

    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): Model output logits of shape (B, C, H, W) or (B, 1, H, W).
            targets (torch.Tensor): Ground truth masks of shape (B, C, H, W) or (B, 1, H, W).
        """
        # Apply sigmoid to convert logits to probabilities
        probs = torch.sigmoid(logits)

        # Flatten the spatial dimensions: (B, C, H, W) -> (B, -1)
        # We assume C=1 for binary segmentation, so we just flatten everything after batch dim
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)

        # Calculate intersection and union per image
        intersection = (probs_flat * targets_flat).sum(dim=1)
        union = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)

        # Calculate Dice score
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Return mean loss over the batch
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    """
    Combines Binary Cross Entropy (BCE) and Dice Loss.
    This is the core objective function for the segmentation task.
    """

    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1.0):
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(smooth=smooth)

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): Model output logits.
            targets (torch.Tensor): Ground truth masks.
        """
        # Ensure targets are float for BCE calculation
        if targets.dtype != torch.float32:
            targets = targets.float()

        # Calculate individual losses
        bce = self.bce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)

        # Weighted sum
        return (self.bce_weight * bce) + (self.dice_weight * dice)


class DeepSupervisionLoss(nn.Module):
    """
    Wrapper for Deep Supervision.
    Calculates the weighted sum of losses for multi-scale model predictions.
    """

    def __init__(self, weights=None):
        """
        Args:
            weights (list[float], optional): List of weights for each output head.
                                            If None, equal weighting is applied.
        """
        super(DeepSupervisionLoss, self).__init__()
        self.weights = weights
        self.base_loss = BCEDiceLoss()

    def forward(self, predictions, targets):
        """
        Args:
            predictions (list[torch.Tensor] or torch.Tensor): List of model outputs at different scales,
                                                              or a single tensor if deep supervision is off.
            targets (torch.Tensor): Ground truth mask of shape (B, C, H, W).
        """
        # Handle case where model returns a single tensor (inference or no deep supervision)
        if isinstance(predictions, torch.Tensor):
            predictions = [predictions]

        # Ensure targets are 4D: (B, C, H, W)
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)

        # Determine weights
        if self.weights is None:
            # Default to equal weights summing to roughly the number of heads (or just 1.0 each)
            current_weights = [1.0] * len(predictions)
        else:
            current_weights = self.weights
            if len(current_weights) != len(predictions):
                # Fallback if mismatch: use equal weights
                current_weights = [1.0] * len(predictions)

        total_loss = 0.0

        for pred, weight in zip(predictions, current_weights):
            # Check if resizing of target is needed
            # pred shape: (B, C, H_pred, W_pred)
            # targets shape: (B, C, H_gt, W_gt)
            if pred.shape[-2:] != targets.shape[-2:]:
                # Resize ground truth to match prediction resolution.
                # Use 'nearest' interpolation to keep the mask binary (0 or 1).
                target_resized = F.interpolate(
                    targets.float(), size=pred.shape[-2:], mode="nearest"
                )
            else:
                target_resized = targets

            # Calculate loss for this scale
            loss = self.base_loss(pred, target_resized)
            total_loss += weight * loss

        return total_loss
