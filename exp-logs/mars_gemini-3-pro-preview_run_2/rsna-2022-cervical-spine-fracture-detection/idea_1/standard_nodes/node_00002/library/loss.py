import torch
import torch.nn as nn
from library.config import Config


class WeightedLogLoss(nn.Module):
    """
    Weighted Multi-Label Logarithmic Loss for Cervical Spine Fracture Detection.

    This loss function computes the binary cross-entropy between predicted probabilities
    and ground truth labels, weighted by class importance.

    As per the task description:
    - The 'patient_overall' label is weighted more highly than specific vertebrae labels.
    - The loss is averaged across all predictions (rows).
    """

    def __init__(self, overall_weight=7.0, vertebrae_weight=1.0):
        """
        Args:
            overall_weight (float): Weight for the 'patient_overall' class.
            vertebrae_weight (float): Weight for C1-C7 classes.
        """
        super(WeightedLogLoss, self).__init__()

        # Target columns order from dataset.py:
        # ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "patient_overall"]
        # We assign the vertebrae_weight to the first 7 and overall_weight to the last one.
        weights = [vertebrae_weight] * 7 + [overall_weight]

        # register_buffer allows the tensor to be part of the module's state
        # and automatically move to the correct device (CPU/GPU) with the model.
        self.register_buffer(
            "class_weights", torch.tensor(weights, dtype=torch.float32)
        )

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Predicted probabilities of shape (Batch, 8).
                                   Values should be in range [0, 1].
            targets (torch.Tensor): Ground truth binary labels of shape (Batch, 8).

        Returns:
            torch.Tensor: Scalar tensor containing the averaged weighted log loss.
        """
        # Ensure inputs are clamped to avoid log(0) which results in NaN/Inf
        epsilon = 1e-7
        inputs = torch.clamp(inputs, epsilon, 1.0 - epsilon)

        # Compute Binary Cross Entropy element-wise
        # Formula: -[y * log(p) + (1-y) * log(1-p)]
        bce_loss = -(
            targets * torch.log(inputs) + (1.0 - targets) * torch.log(1.0 - inputs)
        )

        # Apply class weights
        # self.class_weights is broadcasted across the batch dimension
        weighted_bce = bce_loss * self.class_weights

        # The metric specifies that loss is averaged across all rows.
        # In this context, a "row" is a single prediction for a specific class and study.
        # So we take the mean over the entire tensor.
        loss = weighted_bce.mean()

        return loss
