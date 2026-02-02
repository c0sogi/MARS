import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class WeightedLabelSmoothCrossEntropy(nn.Module):
    """
    A custom loss module that combines Cross Entropy Loss with:
    1. Class weighting (to handle imbalance).
    2. Label smoothing (to prevent overfitting/overconfidence).

    This implementation wraps torch.nn.functional.cross_entropy which supports
    both features natively.
    """

    def __init__(self, weight=None, smoothing=Config.LABEL_SMOOTHING):
        """
        Args:
            weight (torch.Tensor or list, optional): Pre-calculated class weights.
                                                     Shape should be (C,).
                                                     If provided, it is registered as a buffer.
            smoothing (float): Label smoothing factor (epsilon).
                               Defaults to Config.LABEL_SMOOTHING (0.1).
        """
        super(WeightedLabelSmoothCrossEntropy, self).__init__()
        self.smoothing = smoothing

        # Register weight as a buffer so it is saved with the model
        # and automatically moved to the correct device (CPU/GPU).
        if weight is not None:
            # Ensure weights are float32 as required by cross_entropy
            self.register_buffer("weight", torch.as_tensor(weight, dtype=torch.float32))
        else:
            self.weight = None

    def forward(self, inputs, targets):
        """
        Computes the weighted label-smoothed cross-entropy loss.

        Args:
            inputs (torch.Tensor): Logits from the model. Shape (N, C).
            targets (torch.Tensor): Ground truth. Can be class indices (N,)
                                    or class probabilities/one-hot (N, C).

        Returns:
            torch.Tensor: Scalar loss value (averaged over batch by default).
        """
        return F.cross_entropy(
            inputs, targets, weight=self.weight, label_smoothing=self.smoothing
        )
