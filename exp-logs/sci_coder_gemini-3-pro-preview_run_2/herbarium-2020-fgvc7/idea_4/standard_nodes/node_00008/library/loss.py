import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Implementation of Focal Loss for multi-class classification.
    Formula: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        """
        Args:
            gamma (float): Focusing parameter. Higher values focus more on hard examples.
            alpha (float, list, np.ndarray, or torch.Tensor, optional): Weighting factor.
                If alpha is a list/array/tensor, it is treated as class weights (one per class).
                If alpha is a float, it is applied as a constant scalar.
                If None, no weighting is applied.
            reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

        # Handle alpha initialization
        if isinstance(alpha, (list, tuple)):
            self.alpha = torch.tensor(alpha).float()
        elif isinstance(alpha, (float, int)):
            self.alpha = torch.tensor(alpha).float()
        elif isinstance(alpha, torch.Tensor):
            self.alpha = alpha.float()
        else:
            self.alpha = None

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Logits of shape (Batch, Num_Classes).
            targets (torch.Tensor): Ground truth labels of shape (Batch).

        Returns:
            torch.Tensor: Calculated loss.
        """
        # Compute log probabilities using log_softmax for numerical stability
        log_probs = F.log_softmax(inputs, dim=1)

        # Gather the log probabilities corresponding to the target class
        # targets.view(-1, 1) reshapes to (Batch, 1) for gather
        log_pt = log_probs.gather(1, targets.view(-1, 1))
        log_pt = log_pt.view(-1)  # Reshape back to (Batch)

        # Compute probabilities (p_t)
        pt = log_pt.exp()

        # Compute the focal term: (1 - p_t)^gamma
        focal_term = (1 - pt).pow(self.gamma)

        # Compute the basic loss: -log(p_t)
        loss = -log_pt * focal_term

        # Apply alpha weighting if provided
        if self.alpha is not None:
            # Move alpha to the correct device if necessary
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)

            if self.alpha.dim() == 0:
                # Scalar alpha
                loss = loss * self.alpha
            else:
                # Class-wise weights
                # Gather alpha values corresponding to the targets
                at = self.alpha.gather(0, targets)
                loss = loss * at

        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
