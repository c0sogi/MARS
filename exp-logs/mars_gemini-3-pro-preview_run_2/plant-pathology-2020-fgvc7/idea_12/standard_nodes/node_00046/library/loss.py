import torch
import torch.nn as nn


class WeightedBCELoss(nn.Module):
    """
    Custom Loss function for Apple Disease Detection.

    Wraps BCEWithLogitsLoss to provide:
    1. Inverse Class Frequency Weighting (via pos_weight).
    2. Label Smoothing for regularization.
    """

    def __init__(self, pos_weights: torch.Tensor = None, smoothing: float = 0.05):
        """
        Args:
            pos_weights (torch.Tensor, optional): Weights for positive examples for each class.
                                                  Shape: [num_classes].
            smoothing (float): Label smoothing factor. Default is 0.05.
                               Targets are transformed: y_new = y * (1 - alpha) + 0.5 * alpha
        """
        super(WeightedBCELoss, self).__init__()
        self.smoothing = smoothing

        # Initialize the base criterion
        # pos_weight allows trading off recall and precision by up-weighting the positive class.
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the loss function.

        Args:
            inputs (torch.Tensor): Logits from the model. Shape: [batch_size, num_classes].
            targets (torch.Tensor): Binary targets. Shape: [batch_size, num_classes].

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Apply Label Smoothing if enabled
        if self.smoothing > 0.0:
            with torch.no_grad():
                # For binary targets (0 or 1), smoothing pushes them towards 0.5
                # 1 becomes (1 - smoothing) + 0.5 * smoothing
                # 0 becomes 0.5 * smoothing
                targets_smooth = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing
        else:
            targets_smooth = targets

        # Compute BCE Loss
        return self.criterion(inputs, targets_smooth)
