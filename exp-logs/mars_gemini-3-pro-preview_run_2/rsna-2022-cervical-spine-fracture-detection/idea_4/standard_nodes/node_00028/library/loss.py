import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class WeightedFractureLoss(nn.Module):
    """
    Weighted Multi-Label Logarithmic Loss for Cervical Spine Fracture Detection.

    Features:
    1. Competition Weights: Prioritizes 'patient_overall' (weight 1.0) vs individual
       vertebrae (weight 1/7), as per the task description.
    2. Positive Class Weighting: Applies a fixed positive weight (from Config) to
       fracture targets to improve sensitivity and handle class imbalance.
    """

    def __init__(self):
        super(WeightedFractureLoss, self).__init__()

        # Define competition weights based on Config.TARGET_COLS order:
        # ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "patient_overall"]
        # C1-C7 get weight 1/7 (~0.142), patient_overall gets weight 1.0.
        # This balances the contribution of the specific locations with the global outcome.
        comp_weights = [1.0 / 7.0] * 7 + [1.0]

        # Define positive class weights for sensitivity (e.g., 2.5)
        # Applied to all classes equally to penalize false negatives.
        pos_weights = [Config.POS_WEIGHT] * Config.NUM_CLASSES

        # Register as buffers so they are saved with state_dict and move to device automatically
        self.register_buffer("competition_weights", torch.tensor(comp_weights))
        self.register_buffer("pos_weights", torch.tensor(pos_weights))

    def forward(self, logits, targets):
        """
        Calculate the weighted loss.

        Args:
            logits (torch.Tensor): Predicted logits of shape (Batch, 8).
            targets (torch.Tensor): Ground truth labels of shape (Batch, 8).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Calculate Binary Cross Entropy with Logits
        # pos_weight argument handles the class imbalance (sensitivity)
        # reduction='none' preserves the shape (Batch, 8) so we can apply competition weights
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets.float(), pos_weight=self.pos_weights, reduction="none"
        )

        # Apply competition-specific weights (broadcasting over batch dimension)
        weighted_loss = bce_loss * self.competition_weights

        # Return the mean loss over the batch and all labels
        return weighted_loss.mean()
