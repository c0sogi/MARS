import torch
import torch.nn as nn
from library.config import Config


class WeightedMultiLabelLoss(nn.Module):
    """
    Custom loss function for Cervical Spine Fracture Detection.

    Implements a weighted binary cross-entropy loss that addresses:
    1. Class Imbalance: Using `pos_weight` to penalize false negatives more heavily (sensitivity).
    2. Competition Metric: Using `weight` (class_weights) to assign higher importance to
       specific labels (e.g., patient_overall) as per the task description.
    """

    def __init__(self, pos_weight_value=None, class_weights=None):
        """
        Args:
            pos_weight_value (float, optional): Weight for the positive class (1).
                                              Defaults to Config.POS_WEIGHT.
            class_weights (list/tensor, optional): Weights for each of the 8 classes.
                                                 If None, uniform weights are used.
        """
        super().__init__()

        # 1. Positive Class Weighting (for Sensitivity)
        # Addresses the high imbalance between fractured (1) and healthy (0) vertebrae.
        # This corresponds to the 'pos_weight' argument in BCEWithLogitsLoss.
        if pos_weight_value is None:
            pos_weight_value = Config.POS_WEIGHT

        # Create a tensor of shape [NUM_CLASSES] filled with the pos_weight value.
        # We register it as a buffer to ensure it moves to the correct device
        # (CPU/GPU) automatically if the module is moved.
        self.register_buffer(
            "pos_weight", torch.full((Config.NUM_CLASSES,), pos_weight_value)
        )

        # 2. Class-Specific Weighting (for Metric Alignment)
        # Corresponds to w_j in the competition metric L_ij = -w_j * [...]
        # This corresponds to the 'weight' argument in BCEWithLogitsLoss.
        if class_weights is not None:
            self.register_buffer(
                "class_weights", torch.tensor(class_weights, dtype=torch.float32)
            )
        else:
            self.class_weights = None

    def forward(self, logits, targets):
        """
        Calculates the weighted BCE loss.

        Args:
            logits (torch.Tensor): Raw model predictions of shape (Batch, Num_Classes).
            targets (torch.Tensor): Ground truth labels of shape (Batch, Num_Classes).

        Returns:
            torch.Tensor: Scalar loss value (averaged over batch).
        """
        # BCEWithLogitsLoss combines Sigmoid and BCE for numerical stability.
        # pos_weight: Weight for positive examples (sensitivity).
        # weight: Rescaling weight for each class (competition metric w_j).
        loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=self.pos_weight, weight=self.class_weights, reduction="mean"
        )

        # Ensure targets are float for BCE calculation
        return loss_fn(logits, targets.float())
