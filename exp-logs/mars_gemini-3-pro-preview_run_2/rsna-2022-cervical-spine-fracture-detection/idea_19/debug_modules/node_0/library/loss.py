import torch
import torch.nn as nn
from library.config import Config


class WeightedMultiLabelLoss(nn.Module):
    """
    Weighted Multi-Label Logarithmic Loss.

    Computes the binary cross-entropy for each of the 8 targets (C1-C7, patient_overall),
    weights them according to the competition metric, and averages across the batch.

    The weights are normalized to sum to 1.0 to ensure the loss scale is consistent.
    Positive class weighting is strictly avoided to maintain probability calibration.
    """

    def __init__(self):
        super().__init__()

        # Load weights from Config
        # Config.CLASS_WEIGHTS is expected to be [1/14, 1/14, ..., 7/14]
        weights_list = Config.CLASS_WEIGHTS

        # Convert to tensor
        weights = torch.tensor(weights_list, dtype=torch.float32)

        # Normalize weights to sum to 1.0
        # This ensures that the sum of weighted losses represents a proper weighted average
        total_weight = weights.sum()
        if total_weight > 0:
            weights = weights / total_weight

        # Register weights as a buffer so they are part of the state_dict
        # and automatically move to the correct device (CPU/GPU) with the module.
        self.register_buffer("class_weights", weights)

        # Base Loss Function
        # reduction='none' is used to apply class-specific weights manually after calculation.
        # pos_weight is intentionally NOT used (default 1.0) to prevent probability drift
        # and ensure the model outputs calibrated probabilities required for Log Loss.
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculates the weighted loss.

        Args:
            logits (torch.Tensor): Raw model outputs (before sigmoid) of shape (Batch_Size, 8).
            targets (torch.Tensor): Binary ground truth labels of shape (Batch_Size, 8).

        Returns:
            torch.Tensor: Scalar tensor representing the mean weighted loss over the batch.
        """
        # Ensure targets have the same floating point type as logits
        targets = targets.type_as(logits)

        # 1. Calculate Binary Cross Entropy for each element
        # Shape: (Batch_Size, Num_Classes)
        raw_loss = self.bce_loss(logits, targets)

        # 2. Apply Class Weights
        # Shape: (Batch_Size, Num_Classes) * (Num_Classes,) -> (Batch_Size, Num_Classes)
        # Broadcasting automatically aligns the weights to the class dimension.
        weighted_loss = raw_loss * self.class_weights

        # 3. Aggregate Loss
        # First, sum over the classes (dim=1) to get the total weighted loss per study.
        # Since weights sum to 1.0, this is the weighted average loss for that specific exam.
        study_loss = weighted_loss.sum(dim=1)

        # Finally, average over the batch (dim=0) to get the scalar loss for optimization.
        final_loss = study_loss.mean()

        return final_loss
