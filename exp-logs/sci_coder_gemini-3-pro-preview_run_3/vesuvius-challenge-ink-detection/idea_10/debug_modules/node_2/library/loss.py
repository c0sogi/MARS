import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DiceBCELoss(nn.Module):
    """
    Balanced Loss combining Binary Cross Entropy and Dice Loss.
    Used for the primary ink segmentation task.
    """

    def __init__(self, smooth: float = 1e-6):
        super(DiceBCELoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits from the model (B, C, H, W) or (B, H, W).
            targets: Binary ground truth mask (same shape as inputs).

        Returns:
            Combined BCE + Dice loss.
        """
        # Flatten inputs and targets
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # 1. BCE Loss (with logits for numerical stability)
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="mean")

        # 2. Dice Loss
        probs = torch.sigmoid(inputs)
        intersection = (probs * targets).sum()
        union = probs.sum() + targets.sum()

        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice_score

        # Combine
        return bce_loss + dice_loss


class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross Entropy Loss to handle class imbalance.
    Used for the auxiliary boundary detection task where edges are sparse.

    If pos_weight is not provided, it is calculated dynamically per batch
    as (number_of_negatives / number_of_positives).
    """

    def __init__(self, pos_weight: float = None):
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits from the model.
            targets: Binary ground truth mask.

        Returns:
            Weighted BCE loss.
        """
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        if self.pos_weight is not None:
            weight = torch.tensor(self.pos_weight, device=inputs.device)
        else:
            # Dynamic weighting based on current batch statistics
            # Avoid division by zero
            n_pos = targets.sum()
            n_neg = targets.numel() - n_pos
            if n_pos > 0:
                weight = n_neg / n_pos
            else:
                weight = torch.tensor(1.0, device=inputs.device)

        # BCEWithLogitsLoss accepts pos_weight
        return F.binary_cross_entropy_with_logits(
            inputs, targets, pos_weight=weight, reduction="mean"
        )


class JointLoss(nn.Module):
    """
    Joint objective function optimizing both mask and boundary predictions.
    L_total = L_mask + lambda * L_boundary
    """

    def __init__(self):
        super(JointLoss, self).__init__()
        self.mask_loss_fn = DiceBCELoss()
        self.boundary_loss_fn = WeightedBCELoss()
        self.aux_weight = Config.AUX_LOSS_WEIGHT

    def forward(self, outputs: dict, targets: dict) -> torch.Tensor:
        """
        Args:
            outputs: Dictionary containing model predictions:
                     {'mask': Tensor, 'boundary': Tensor}
            targets: Dictionary containing ground truth:
                     {'mask': Tensor, 'boundary': Tensor}

        Returns:
            Total combined loss.
        """
        # Primary Mask Loss
        mask_pred = outputs["mask"]
        mask_target = targets["mask"]
        l_mask = self.mask_loss_fn(mask_pred, mask_target)

        # Auxiliary Boundary Loss
        boundary_pred = outputs["boundary"]
        boundary_target = targets["boundary"]
        l_boundary = self.boundary_loss_fn(boundary_pred, boundary_target)

        # Total Loss
        total_loss = l_mask + (self.aux_weight * l_boundary)

        return total_loss
