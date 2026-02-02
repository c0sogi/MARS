import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedSoftTargetCrossEntropy(nn.Module):
    """
    Weighted Soft-Target Cross-Entropy Loss.

    This loss function accepts floating-point probability targets (soft targets) instead of
    hard integer indices. This preserves label uncertainty and prevents the model from
    becoming overconfident. It also supports class weighting to handle class imbalance.

    Formula:
        loss = - sum(weights * targets * log_softmax(inputs)) / N
    """

    def __init__(self, weight=None, reduction="mean"):
        """
        Args:
            weight (torch.Tensor, optional): A manual rescaling weight given to each class.
                If given, has to be a Tensor of size C. Defaults to None.
            reduction (str, optional): Specifies the reduction to apply to the output:
                'none' | 'mean' | 'sum'. 'mean': the sum of the output will be divided by
                the number of elements in the output, 'sum': the output will be summed.
                Defaults to 'mean'.
        """
        super(WeightedSoftTargetCrossEntropy, self).__init__()
        self.reduction = reduction

        # Register weight as a buffer so it is part of the state_dict and moves to device
        if weight is not None:
            self.register_buffer("weight", weight)
        else:
            self.weight = None

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Predictions (logits) of shape (Batch_Size, Num_Classes).
            targets (torch.Tensor): Soft targets (probabilities) of shape (Batch_Size, Num_Classes).

        Returns:
            torch.Tensor: The computed loss.
        """
        # Compute log probabilities from logits for numerical stability
        log_probs = F.log_softmax(inputs, dim=1)

        # Compute the element-wise cross-entropy term: - targets * log(predictions)
        # Shape: (Batch_Size, Num_Classes)
        loss = -targets * log_probs

        # Apply class weights if provided
        if self.weight is not None:
            # weight shape (C,) broadcasts to (Batch_Size, C)
            loss = loss * self.weight

        # Sum over the class dimension (dim=1) to get the loss per sample
        # Shape: (Batch_Size,)
        loss = loss.sum(dim=1)

        # Apply the specified reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
