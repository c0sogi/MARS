import os
import pandas as pd
import torch
import torch.nn as nn
from library.config import Config


class WeightedBCELoss(nn.Module):
    """
    Custom Binary Cross Entropy Loss with inverse class frequency weighting.
    Designed to handle the class imbalance in the Right Whale Detection task.

    This loss wraps nn.BCEWithLogitsLoss and automatically calculates the
    pos_weight parameter based on the ratio of negative to positive samples
    in the training metadata.
    """

    def __init__(self, device=Config.DEVICE):
        """
        Initialize the WeightedBCELoss.

        Args:
            device (str): The device to place the weight tensor on. Defaults to Config.DEVICE.
        """
        super().__init__()
        self.device = device
        self.use_weighted = Config.USE_WEIGHTED_LOSS

        # Calculate positive weight based on training data distribution
        self.pos_weight_val = self._calculate_pos_weight()

        if self.use_weighted:
            # BCEWithLogitsLoss expects pos_weight as a Tensor.
            # It should be broadcastable to the target shape.
            # For binary classification with shape (N, 1), a 1D tensor of size 1 is appropriate.
            pos_weight_tensor = torch.tensor([self.pos_weight_val], device=self.device)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        else:
            self.criterion = nn.BCEWithLogitsLoss()

    def _calculate_pos_weight(self):
        """
        Calculates pos_weight = number_of_negatives / number_of_positives
        based on the training metadata.

        Returns:
            float: The calculated positive weight. Returns 1.0 if calculation fails.
        """
        if not os.path.exists(Config.TRAIN_CSV):
            # Fallback if metadata is missing
            return 1.0

        try:
            df = pd.read_csv(Config.TRAIN_CSV)
            if "label" not in df.columns:
                return 1.0

            counts = df["label"].value_counts()

            n_pos = counts.get(1, 0)
            n_neg = counts.get(0, 0)

            if n_pos == 0:
                return 1.0

            # Inverse class frequency weighting for the positive class
            # pos_weight > 1 increases the penalty for False Negatives (missing a whale call)
            weight = n_neg / n_pos
            return weight

        except Exception:
            return 1.0

    def forward(self, inputs, targets):
        """
        Compute the loss.

        Args:
            inputs (torch.Tensor): Logits from the model (N, 1).
            targets (torch.Tensor): Ground truth labels (N, 1) or (N,).

        Returns:
            torch.Tensor: The computed loss.
        """
        # Ensure targets are float for BCE
        targets = targets.float()

        # Reshape targets to match inputs (N, 1) if necessary
        # This prevents broadcasting errors if targets are passed as a flat vector (N,)
        if targets.ndim == 1:
            targets = targets.view(-1, 1)

        # Double check shape consistency
        if inputs.shape != targets.shape:
            targets = targets.view_as(inputs)

        return self.criterion(inputs, targets)
