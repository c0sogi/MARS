import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class WeightedMultiLabelLogLoss(nn.Module):
    """
    Weighted Multi-Label Logarithmic Loss.

    This loss function implements the specific metric used in the competition:
    L_ij = -w_j * [y_ij * log(p_ij) + (1 - y_ij) * log(1 - p_ij)]

    Where:
        - p_ij is the predicted probability (sigmoid of logit).
        - y_ij is the ground truth label.
        - w_j is the weight for the specific class j (vertebrae or patient_overall).

    The final loss is averaged across all rows (i.e., all samples in the batch and all class columns),
    consistent with the competition's evaluation method.
    """

    def __init__(self):
        super(WeightedMultiLabelLogLoss, self).__init__()
        # Load weights from Config.
        # Config.LOSS_WEIGHTS is expected to be a list like [1.0, ..., 7.0]
        # We convert it to a tensor and reshape to (1, -1) to broadcast over the batch dimension.
        self.weights = torch.tensor(Config.LOSS_WEIGHTS, dtype=torch.float32).view(
            1, -1
        )

    def forward(self, logits, targets):
        """
        Computes the weighted binary cross entropy loss.

        Args:
            logits (torch.Tensor): Raw model outputs (before sigmoid) of shape (Batch, Num_Classes).
            targets (torch.Tensor): Ground truth binary labels of shape (Batch, Num_Classes).

        Returns:
            torch.Tensor: The scalar loss value averaged across all predictions.
        """
        # Ensure weights are on the same device as the input logits
        device = logits.device
        weights = self.weights.to(device)

        # Ensure targets are float32 for BCE calculation
        targets = targets.to(dtype=torch.float32)

        # Compute Binary Cross Entropy with Logits.
        # We use the 'weight' parameter to apply the class-specific weights w_j.
        # We do NOT use 'pos_weight' to ensure probabilistic calibration is maintained.
        # reduction='mean' ensures the loss is averaged across all elements (rows in submission).
        loss = F.binary_cross_entropy_with_logits(
            input=logits, target=targets, weight=weights, reduction="mean"
        )

        return loss
