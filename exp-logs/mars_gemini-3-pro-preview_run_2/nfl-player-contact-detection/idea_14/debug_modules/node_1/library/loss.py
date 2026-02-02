import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Implementation of Focal Loss for binary classification.

    Formula:
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    where p_t is the model's estimated probability for the target class.

    Args:
        alpha (float): Weighting factor for the positive class (0 < alpha < 1).
                       The negative class will be weighted by (1 - alpha).
                       Default: 0.25 (common for high imbalance like RetinaNet).
        gamma (float): Focusing parameter to down-weight easy examples.
                       Default: 2.0.
        reduction (str): Specifies the reduction to apply to the output:
                         'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Raw logits from the model (no sigmoid applied).
                                   Shape: (batch_size, ) or (batch_size, 1)
            targets (torch.Tensor): Binary ground truth labels (0 or 1).
                                    Shape: same as inputs
        Returns:
            torch.Tensor: The computed loss.
        """
        # Ensure inputs and targets are the same shape and type
        if inputs.shape != targets.shape:
            targets = targets.view_as(inputs)

        targets = targets.float()

        # Compute binary cross entropy loss
        # reduction='none' is essential so we can weight individual samples
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

        # Calculate probabilities (p_t)
        # p = sigmoid(inputs)
        # If target=1, p_t = p. If target=0, p_t = 1 - p.
        # However, we can use the property that exp(-bce_loss) = p_t
        p_t = torch.exp(-bce_loss)

        # Calculate the focal term: (1 - p_t)^gamma
        focal_term = (1 - p_t) ** self.gamma

        # Calculate alpha weighting
        # alpha_t = alpha if target=1 else (1 - alpha)
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
