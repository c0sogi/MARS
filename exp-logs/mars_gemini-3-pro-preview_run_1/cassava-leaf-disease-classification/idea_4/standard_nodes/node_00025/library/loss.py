import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftTargetCrossEntropy(nn.Module):
    """
    Cross Entropy Loss that handles continuous/soft targets.
    This is required when using MixUp or CutMix, where the targets are
    linear combinations of one-hot vectors (probabilities) rather than
    discrete class indices.
    """

    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Predicted logits of shape (N, C).
            target (torch.Tensor): Soft targets of shape (N, C).

        Returns:
            torch.Tensor: The mean loss value.
        """
        # Calculate log probabilities from logits
        log_probs = F.log_softmax(x, dim=-1)

        # Compute cross entropy: -sum(p(x) * log(q(x)))
        # Sum over classes (dim=-1), then mean over batch
        loss = torch.sum(-target * log_probs, dim=-1)

        return loss.mean()
