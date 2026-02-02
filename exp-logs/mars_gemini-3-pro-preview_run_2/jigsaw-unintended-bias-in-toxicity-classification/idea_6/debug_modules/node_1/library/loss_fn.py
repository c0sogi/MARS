import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class WeightedMultiTaskLoss(nn.Module):
    """
    Custom Loss function for Identity-Weighted Multi-Task Learning.

    Computes:
    1. Weighted Binary Cross Entropy for the primary toxicity task.
       - Weights are provided per-sample (sample_weights) to prioritize hard examples/identities.
    2. Binary Cross Entropy for the auxiliary identity prediction task.
       - Scaled by a hyperparameter (lambda) to control its influence.

    Total Loss = mean(BCE(tox) * sample_weight) + lambda * mean(BCE(identities))
    """

    def __init__(self, aux_loss_weight=Config.AUX_LOSS_WEIGHT):
        super(WeightedMultiTaskLoss, self).__init__()
        self.aux_loss_weight = aux_loss_weight

    def forward(
        self, toxicity_logits, identity_logits, targets, aux_targets, sample_weights
    ):
        """
        Args:
            toxicity_logits (torch.Tensor): Logits from toxicity head (Batch, 1).
            identity_logits (torch.Tensor): Logits from identity head (Batch, Num_Aux).
            targets (torch.Tensor): Toxicity targets (Batch,).
            aux_targets (torch.Tensor): Identity targets (Batch, Num_Aux).
            sample_weights (torch.Tensor): Per-sample weights (Batch,).

        Returns:
            torch.Tensor: The combined scalar loss.
        """
        # ----------------------------------------------------------------------
        # 1. Primary Task Loss (Toxicity)
        # ----------------------------------------------------------------------
        # Ensure targets and weights match logit dimensions (Batch, 1)
        targets = targets.view(-1, 1)
        sample_weights = sample_weights.view(-1, 1)

        # Compute per-sample BCE loss
        # We use binary_cross_entropy_with_logits which combines Sigmoid + BCE
        # reduction='none' allows us to apply sample weights manually
        tox_loss_per_sample = F.binary_cross_entropy_with_logits(
            toxicity_logits, targets, reduction="none"
        )

        # Apply sample weights
        # This boosts the gradient signal for examples mentioning identities
        weighted_tox_loss = tox_loss_per_sample * sample_weights

        # Average over the batch
        primary_loss = weighted_tox_loss.mean()

        # ----------------------------------------------------------------------
        # 2. Auxiliary Task Loss (Identity Prediction)
        # ----------------------------------------------------------------------
        # Standard multi-label classification loss
        aux_loss = F.binary_cross_entropy_with_logits(
            identity_logits, aux_targets, reduction="mean"
        )

        # ----------------------------------------------------------------------
        # 3. Combine Losses
        # ----------------------------------------------------------------------
        total_loss = primary_loss + (self.aux_loss_weight * aux_loss)

        return total_loss
