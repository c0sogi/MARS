import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class WeightedMultiLabelLogLoss(nn.Module):
    """
    Weighted Multi-label Logarithmic Loss.

    This loss function combines Binary Cross Entropy with Logits, applying specific
    weights to different target columns (vertebrae vs. patient overall) and a
    positive class weight to handle label imbalance.

    Formula:
    L_ij = -w_j * [pos_weight * y_ij * log(p_ij) + (1 - y_ij) * log(1 - p_ij)]
    """

    def __init__(self, pos_weight=None, class_weights=None):
        """
        Args:
            pos_weight (float, optional): Weight for the positive class to handle
                0/1 imbalance. Defaults to Config.POS_WEIGHT.
            class_weights (list or torch.Tensor, optional): Weights for each of the
                8 target columns (C1-C7, Patient Overall). Defaults to
                [1, 1, 1, 1, 1, 1, 1, 7] to weight the overall outcome higher.
        """
        super(WeightedMultiLabelLogLoss, self).__init__()

        # 1. Positive Class Weight (Sensitivity)
        # This scales the loss for positive targets (y=1) to address rarity of fractures.
        val_pos = pos_weight if pos_weight is not None else Config.POS_WEIGHT
        self.register_buffer("pos_weight", torch.tensor(val_pos, dtype=torch.float32))

        # 2. Column/Class Weights (Importance)
        # This scales the loss for specific columns (w_j in the metric formula).
        # Indices 0-6 are C1-C7, Index 7 is patient_overall.
        if class_weights is None:
            # Default heuristic: Weight 'patient_overall' equal to the sum of vertebrae
            # or simply "more highly" as requested.
            w = [1.0] * 7 + [7.0]
        else:
            w = class_weights

        self.register_buffer("class_weights", torch.tensor(w, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes the weighted binary cross entropy loss.

        Args:
            logits (torch.Tensor): Predicted logits of shape (Batch, 8).
            targets (torch.Tensor): Ground truth binary labels of shape (Batch, 8).

        Returns:
            torch.Tensor: Scalar loss value (averaged).
        """
        # F.binary_cross_entropy_with_logits applies sigmoid internally for stability.
        # 'weight' argument applies to the batch element/column (our class_weights).
        # 'pos_weight' argument applies to the positive class terms.
        loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            weight=self.class_weights,
            pos_weight=self.pos_weight,
            reduction="mean",
        )

        return loss
