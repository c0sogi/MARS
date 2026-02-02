import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftTargetCrossEntropy(nn.Module):
    """
    Soft Target Cross Entropy Loss.

    This loss function is designed for training with data augmentation techniques
    like MixUp and CutMix, which generate continuous (soft) targets instead of
    discrete class indices.

    It computes the cross entropy between the input logits and the target
    probability distribution.

    Formula:
        loss = - sum(target * log(softmax(input))) / batch_size
    """

    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes the loss.

        Args:
            x (torch.Tensor): Model predictions (logits) of shape (Batch, NumClasses).
            target (torch.Tensor): Soft targets (probabilities) of shape (Batch, NumClasses).
                                   If targets are indices, they should be one-hot encoded
                                   before passing to this function, or use nn.CrossEntropyLoss.

        Returns:
            torch.Tensor: The calculated scalar loss (averaged over the batch).
        """
        # Compute log probabilities from logits using log_softmax for numerical stability
        log_probs = F.log_softmax(x, dim=-1)

        # Compute the cross entropy: - sum(p(x) * log(q(x)))
        # Sum across the class dimension (dim=-1)
        loss = -torch.sum(target * log_probs, dim=-1)

        # Return the mean loss over the batch
        return loss.mean()
