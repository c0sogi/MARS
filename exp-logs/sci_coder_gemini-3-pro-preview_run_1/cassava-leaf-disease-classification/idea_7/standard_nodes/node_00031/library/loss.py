import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftTargetCrossEntropy(nn.Module):
    """
    Cross Entropy Loss that accepts soft targets (probabilities) instead of hard labels.
    Required for training with MixUp and CutMix strategies where targets are
    linear combinations of one-hot vectors.

    Formula: Loss = - sum(target * log_softmax(input)) / batch_size
    """

    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Predicted logits of shape (N, C)
            target (torch.Tensor): Soft target probabilities of shape (N, C)

        Returns:
            torch.Tensor: Scalar loss value (mean over batch)
        """
        # Compute log probabilities
        log_probs = F.log_softmax(x, dim=-1)

        # Compute cross entropy: - sum(p(x) * log(q(x)))
        # Sum over classes (dim=-1)
        loss = torch.sum(-target * log_probs, dim=-1)

        # Return mean over the batch
        return loss.mean()
