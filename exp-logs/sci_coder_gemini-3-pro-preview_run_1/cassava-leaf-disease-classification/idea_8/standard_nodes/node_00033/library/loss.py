import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftTargetCrossEntropy(nn.Module):
    """
    Cross Entropy Loss for soft targets (e.g., resulting from MixUp or CutMix).

    Standard nn.CrossEntropyLoss typically expects class indices. When using MixUp/CutMix,
    the targets become a probability distribution (e.g., 0.8 for class A, 0.2 for class B).
    This module computes the cross entropy between the input logits and the target probabilities.

    Formula: Loss = - sum(target * log_softmax(input)) / batch_size
    """

    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Predicted logits from the model. Shape (N, C).
            target (torch.Tensor): Soft targets. Shape (N, C).

        Returns:
            torch.Tensor: Scalar loss value (mean over the batch).
        """
        # Compute log probabilities
        logprobs = F.log_softmax(x, dim=-1)

        # Compute cross entropy: - sum(p(x) * log(q(x)))
        # We sum over the class dimension (dim=-1) and then mean over the batch
        loss = -torch.sum(target * logprobs, dim=-1)

        return loss.mean()
