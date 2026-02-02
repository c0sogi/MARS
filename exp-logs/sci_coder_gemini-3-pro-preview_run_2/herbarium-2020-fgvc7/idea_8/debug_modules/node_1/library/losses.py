import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Implementation of Focal Loss for multi-class classification.

    Formula:
        FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

    Args:
        gamma (float): Focusing parameter. Higher values down-weight easy examples.
                       Default: 2.0
        alpha (float, torch.Tensor, or None): Balancing parameter.
                       If float, applies a constant weight.
                       If Tensor, must be of size (C,) containing weights for each class.
                       If None, no alpha weighting is applied.
                       Default: None
        reduction (str): Specifies the reduction to apply to the output:
                         'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
                         'mean': the sum of the output will be divided by the number of
                         elements in the output, 'sum': the output will be summed.
                         Default: 'mean'
    """

    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

        if isinstance(self.alpha, (float, int)):
            self.alpha = torch.tensor([self.alpha])

        # If alpha is a tensor, we don't move it to device here because
        # we don't know the device of inputs yet. We'll handle it in forward.

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Predictions of shape (N, C) where C is the number of classes.
                                   These should be raw logits (before softmax).
            targets (torch.Tensor): Ground truth labels of shape (N,).

        Returns:
            torch.Tensor: The calculated loss.
        """
        # Calculate standard cross entropy loss without reduction
        # F.cross_entropy takes logits and computes log_softmax internally
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")

        # Get the probability of the true class: p_t = exp(-ce_loss)
        pt = torch.exp(-ce_loss)

        # Calculate the focal term: (1 - p_t)^gamma
        focal_term = (1 - pt) ** self.gamma

        # Calculate the base focal loss
        loss = focal_term * ce_loss

        # Apply alpha weighting if provided
        if self.alpha is not None:
            # Ensure alpha is on the correct device and type
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            if self.alpha.dtype != inputs.dtype:
                self.alpha = self.alpha.to(inputs.dtype)

            if self.alpha.numel() == 1:
                # Scalar alpha
                loss = self.alpha * loss
            else:
                # Class-specific weights
                # Gather alpha values corresponding to the target classes
                alpha_t = self.alpha[targets]
                loss = loss * alpha_t

        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none":
            return loss
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")
