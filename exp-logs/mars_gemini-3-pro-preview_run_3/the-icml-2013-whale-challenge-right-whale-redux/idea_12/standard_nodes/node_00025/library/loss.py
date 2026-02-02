import torch
import torch.nn as nn
from library.config import Config


class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross Entropy Loss.

    This module implements Binary Cross-Entropy loss with Logits, applying a
    positive class weight to handle class imbalance (e.g., whale calls vs noise).
    It serves as a wrapper around torch.nn.BCEWithLogitsLoss with default
    parameters derived from the project configuration.
    """

    def __init__(self, pos_weight=None, reduction="mean"):
        """
        Initialize the WeightedBCELoss module.

        Args:
            pos_weight (float or torch.Tensor, optional): Weight for the positive class.
                If None, the value is taken from Config.POS_WEIGHT.
                This is used to counteract the imbalance between the minority (whale)
                and majority (noise) classes.
            reduction (str, optional): Specifies the reduction to apply to the output:
                'none' | 'mean' | 'sum'. Default: 'mean'.
        """
        super(WeightedBCELoss, self).__init__()

        # Determine the weight to use
        if pos_weight is None:
            weight_val = Config.POS_WEIGHT
        else:
            weight_val = pos_weight

        # Convert to tensor if it is a standard python number
        if not isinstance(weight_val, torch.Tensor):
            weight_tensor = torch.tensor(weight_val, dtype=torch.float32)
        else:
            weight_tensor = weight_val.float()

        # Initialize the underlying BCEWithLogitsLoss
        # Passing pos_weight here registers it as a buffer within the criterion,
        # ensuring it automatically moves to the correct device (CPU/GPU) when
        # .to(device) is called on this module.
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=weight_tensor, reduction=reduction
        )

    def forward(self, inputs, targets):
        """
        Calculate the weighted binary cross entropy loss.

        Args:
            inputs (torch.Tensor): Predictions (logits) from the model.
                                   Shape: (Batch Size, 1) or (Batch Size,).
            targets (torch.Tensor): Ground truth labels.
                                    Shape: (Batch Size, 1) or (Batch Size,).

        Returns:
            torch.Tensor: The calculated loss.
        """
        # Ensure targets are the same data type as inputs (usually float32)
        # BCEWithLogitsLoss requires floating point targets.
        if targets.dtype != inputs.dtype:
            targets = targets.to(inputs.dtype)

        return self.criterion(inputs, targets)
