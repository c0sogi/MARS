import torch
import torch.nn as nn
from library.config import Config


class GlobalBatchDiceLoss(nn.Module):
    """
    Computes the Global Dice Coefficient Loss for a batch of predictions.

    This loss treats the entire batch as a single volume, flattening all samples
    into one large vector. This stabilizes the gradient and metric calculation,
    especially when many samples in the batch are empty (common in contrail detection).

    Formula:
        Loss = 1 - (2 * Intersection + epsilon) / (Cardinality + epsilon)
        where Intersection and Cardinality are summed over the entire batch.
    """

    def __init__(self, smooth=1e-6):
        super(GlobalBatchDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Predicted probabilities (0-1). Shape (B, C, H, W) or (B, H, W).
            targets (torch.Tensor): Ground truth binary masks (0 or 1). Shape (B, C, H, W) or (B, H, W).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Flatten inputs and targets to (N,)
        # This aggregates all pixels from all images in the batch
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (inputs_flat * targets_flat).sum()
        # Cardinality: Sum of probabilities + Sum of true pixels
        cardinality = inputs_flat.sum() + targets_flat.sum()

        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

        return 1.0 - dice_score
