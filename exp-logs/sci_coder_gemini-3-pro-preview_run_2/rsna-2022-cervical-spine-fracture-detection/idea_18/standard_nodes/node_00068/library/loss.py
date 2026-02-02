import torch
import torch.nn as nn
from library.config import Config


class WeightedMultiLabelLoss(nn.Module):
    """
    Implements the Weighted Multi-Label Logarithmic Loss for the Cervical Spine Fracture Detection task.

    This loss function calculates the Binary Cross Entropy (BCE) for each of the 8 target labels
    (C1-C7 and patient_overall). It applies specific weights to each class as defined in the
    configuration (normalized to sum to 1.0) and averages the result across all predictions.

    Crucially, this implementation does NOT use positive class weighting (pos_weight) to ensure
    the predicted probabilities remain calibrated, which is essential for the logarithmic loss metric.
    """

    def __init__(self):
        super(WeightedMultiLabelLoss, self).__init__()

        # Retrieve weights from Config
        # Weights are already normalized in Config to sum to 1.0
        # We clone and detach to ensure it's a clean tensor, then move to the configured device
        self.weights = Config.loss_weights_tensor.clone().detach().to(Config.device)

        # Initialize the base Binary Cross Entropy loss
        # reduction='none' is required to apply class-specific weights manually before aggregation
        # pos_weight is explicitly NOT used to ensure probabilistic calibration
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the loss calculation.

        Args:
            logits (torch.Tensor): Model output logits of shape [batch_size, num_classes].
            targets (torch.Tensor): Ground truth binary labels of shape [batch_size, num_classes].

        Returns:
            torch.Tensor: Scalar loss value representing the mean weighted log loss.
        """
        # Ensure weights are on the correct device (handles cases where inputs might be on a different GPU)
        if self.weights.device != logits.device:
            self.weights = self.weights.to(logits.device)

        # Compute raw binary cross entropy loss per element
        # Cast targets to float as required by BCEWithLogitsLoss
        raw_loss = self.bce(logits, targets.float())

        # Apply class-specific weights
        # self.weights (shape [num_classes]) broadcasts to [batch_size, num_classes]
        weighted_loss = raw_loss * self.weights

        # Calculate the final loss
        # "Finally, loss is averaged across all rows."
        # This implies taking the mean over both the batch dimension and the class dimension.
        loss = weighted_loss.mean()

        return loss
