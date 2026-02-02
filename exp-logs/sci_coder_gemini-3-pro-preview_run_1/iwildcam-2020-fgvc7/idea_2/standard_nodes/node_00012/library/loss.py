import torch
import torch.nn as nn
import torch.nn.functional as F
from library import config


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in multi-class classification.
    Formula: FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(
        self,
        alpha=config.FOCAL_LOSS_ALPHA,
        gamma=config.FOCAL_LOSS_GAMMA,
        reduction="mean",
    ):
        """
        Args:
            alpha (float, torch.Tensor, or None): Weighting factor.
                - If float: Applies a constant scalar scaling (balance factor).
                - If torch.Tensor: Should be of shape [num_classes]. Applies specific weight per class.
                - If None: No alpha weighting is applied.
            gamma (float): Focusing parameter. Higher values focus more on hard examples.
            reduction (str): Specifies the reduction to apply to the output: 'none', 'mean', 'sum'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Predictions (logits) of shape [batch_size, num_classes].
            targets (torch.Tensor): Ground truth labels of shape [batch_size].

        Returns:
            torch.Tensor: The computed loss.
        """
        # 1. Calculate Cross Entropy Loss: -log(pt)
        # reduction='none' is required to apply per-sample modulation
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")

        # 2. Calculate pt (probability of the true class)
        pt = torch.exp(-ce_loss)

        # 3. Calculate the Focal Loss component: (1 - pt)^gamma * ce_loss
        loss = (1 - pt) ** self.gamma * ce_loss

        # 4. Apply Alpha Weighting
        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                # Ensure alpha tensor is on the same device as inputs
                if self.alpha.device != inputs.device:
                    self.alpha = self.alpha.to(inputs.device)

                # Gather the weight corresponding to each target class
                # alpha shape: [num_classes], targets shape: [batch_size]
                alpha_t = self.alpha[targets]
                loss = alpha_t * loss
            elif isinstance(self.alpha, (float, int)):
                # Apply scalar scaling
                loss = self.alpha * loss

        # 5. Apply Reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
