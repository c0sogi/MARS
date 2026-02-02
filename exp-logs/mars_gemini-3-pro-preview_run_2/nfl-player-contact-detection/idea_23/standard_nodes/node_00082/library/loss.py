import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class FocalLoss(nn.Module):
    """
    Binary Focal Loss implementation.

    References:
        Lin et al., https://arxiv.org/abs/1708.02002
    """

    def __init__(
        self,
        alpha: float = Config.FOCAL_ALPHA,
        gamma: float = Config.FOCAL_GAMMA,
        reduction: str = "mean",
    ):
        """
        Args:
            alpha (float): Weighting factor for the positive class (0 < alpha < 1).
                           Class 1 gets weight alpha, Class 0 gets weight (1 - alpha).
                           Default is 0.25 as per Config.
            gamma (float): Focusing parameter to down-weight easy examples.
                           Default is 2.0 as per Config.
            reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs (torch.Tensor): Logits from the model of shape (N, *) or (N, 1).
            targets (torch.Tensor): Ground truth binary labels of shape (N, *) or (N, 1).
                                    Must be the same shape as inputs.

        Returns:
            torch.Tensor: Computed Focal Loss.
        """
        # Ensure targets are float for BCE calculation
        targets = targets.float()

        # Reshape inputs and targets to ensure they match (flattening is usually safest for binary)
        # However, preserving shape allows for 'none' reduction to work as expected on the batch structure
        if inputs.shape != targets.shape:
            # Attempt to broadcast or view if dimensions differ slightly (e.g. (N,1) vs (N,))
            targets = targets.view_as(inputs)

        # Calculate Binary Cross Entropy with Logits (numerically stable)
        # reduction='none' so we can apply focal weights element-wise
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

        # Get the probabilities associated with the true class (p_t)
        # p_t = p if y=1 else 1-p
        # We can calculate p from logits: p = sigmoid(logits)
        p = torch.sigmoid(inputs)
        p_t = p * targets + (1 - p) * (1 - targets)

        # Calculate the focal term: (1 - p_t)^gamma
        focal_term = (1 - p_t) ** self.gamma

        # Calculate the alpha term
        # alpha_t = alpha if y=1 else 1-alpha
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * focal_term * bce_loss
        else:
            loss = focal_term * bce_loss

        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
