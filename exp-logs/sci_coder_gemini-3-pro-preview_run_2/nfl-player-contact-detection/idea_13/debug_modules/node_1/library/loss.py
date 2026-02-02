import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=Config.FOCAL_LOSS_GAMMA, reduction="mean"):
        """
        Initializes the Focal Loss function for binary classification.

        This loss helps address class imbalance by down-weighting easy examples
        and focusing training on hard negatives/positives.

        Args:
            alpha (float, optional): Weighting factor for the positive class (0 < alpha < 1).
                                     If None, no alpha weighting is applied.
                                     Default is 0.25.
            gamma (float, optional): Focusing parameter. Higher values focus more on hard examples.
                                     Default is taken from Config.FOCAL_LOSS_GAMMA.
            reduction (str, optional): Specifies the reduction to apply to the output:
                                       'none' | 'mean' | 'sum'. Default: 'mean'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Computes the focal loss between logits and targets.

        Args:
            inputs (torch.Tensor): Raw logits from the model (before sigmoid).
                                   Shape should match targets.
            targets (torch.Tensor): Binary ground truth labels (0 or 1).
                                    Shape should match inputs.

        Returns:
            torch.Tensor: The computed loss value.
        """
        # Ensure targets are float for BCE calculation
        targets = targets.float()

        # Compute binary cross entropy with logits
        # reduction='none' is required to apply the focal weights element-wise first
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

        # Calculate pt: the probability of the ground truth class
        # Since BCE = -log(pt), we can compute pt as exp(-BCE)
        pt = torch.exp(-bce_loss)

        # Calculate the focal term: (1 - pt)^gamma
        focal_term = (1 - pt) ** self.gamma

        # Apply alpha weighting if specified
        if self.alpha is not None:
            # alpha_t is alpha for class 1 and (1-alpha) for class 0
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
