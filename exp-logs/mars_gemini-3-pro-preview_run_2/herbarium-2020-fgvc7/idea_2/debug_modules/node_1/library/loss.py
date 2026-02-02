import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in classification tasks.
    Formula: FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(
        self, alpha=Config.FOCAL_ALPHA, gamma=Config.FOCAL_GAMMA, reduction="mean"
    ):
        """
        Args:
            alpha (float or torch.Tensor, optional): Weighting factor.
                If float, acts as a constant scalar multiplier.
                If Tensor, must be of shape [num_classes] containing class weights.
                Defaults to Config.FOCAL_ALPHA.
            gamma (float): Focusing parameter. Defaults to Config.FOCAL_GAMMA.
            reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Logits of shape (N, C) where C is number of classes.
            targets (torch.Tensor): Ground truth labels of shape (N).

        Returns:
            torch.Tensor: Computed loss.
        """
        # Compute cross entropy loss (log(pt))
        # reduction='none' ensures we get a loss value per sample
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")

        # Calculate p_t (probability of the true class)
        pt = torch.exp(-ce_loss)

        # Calculate focal term: (1 - p_t)^gamma
        focal_term = (1 - pt) ** self.gamma

        # Calculate basic focal loss
        loss = focal_term * ce_loss

        # Apply alpha weighting
        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                # Ensure alpha is on the same device as inputs
                if self.alpha.device != inputs.device:
                    self.alpha = self.alpha.to(inputs.device)

                # Gather alpha values corresponding to targets
                # self.alpha should be shape [C], alpha_t becomes shape [N]
                alpha_t = self.alpha.gather(0, targets)
                loss = alpha_t * loss
            else:
                # Treat as scalar
                loss = self.alpha * loss

        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
