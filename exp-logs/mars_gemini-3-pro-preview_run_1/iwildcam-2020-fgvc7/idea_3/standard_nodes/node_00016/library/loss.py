import torch
import torch.nn as nn
import torch.nn.functional as F
from library import config


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.

    Formula: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Where:
        p_t is the model's estimated probability for the target class.
        gamma controls the focusing parameter (down-weighting easy examples).
        alpha is a weighting factor for classes (optional).
    """

    def __init__(self, gamma=config.FOCAL_LOSS_GAMMA, alpha=None, reduction="mean"):
        """
        Args:
            gamma (float): Focusing parameter. Higher values down-weight easy examples more.
                           Defaults to config.FOCAL_LOSS_GAMMA.
            alpha (Tensor, optional): Tensor of weights for each class (shape: [num_classes]).
                                      If None, no alpha weighting is applied.
            reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs (Tensor): Predicted logits of shape (N, C).
            targets (Tensor): Ground truth labels of shape (N).

        Returns:
            Tensor: The calculated loss.
        """
        # 1. Calculate standard Cross Entropy Loss (log(p_t))
        # We use reduction='none' to apply focal weights element-wise first
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")

        # 2. Calculate probabilities p_t
        # p_t = exp(-CE)
        pt = torch.exp(-ce_loss)

        # 3. Calculate the focal term: (1 - p_t)^gamma
        focal_term = (1 - pt) ** self.gamma

        # 4. Calculate the base Focal Loss
        loss = focal_term * ce_loss

        # 5. Apply Alpha Weighting (if provided)
        if self.alpha is not None:
            # Ensure alpha is on the correct device
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)

            # Gather the alpha value corresponding to each target class
            # alpha_t shape: (N,)
            alpha_t = self.alpha[targets]

            # Apply alpha weights
            loss = alpha_t * loss

        # 6. Apply Reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
