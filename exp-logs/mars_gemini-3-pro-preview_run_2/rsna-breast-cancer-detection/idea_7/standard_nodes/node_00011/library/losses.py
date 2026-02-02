import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class FocalLoss(nn.Module):
    """
    Binary Focal Loss implementation for addressing class imbalance.

    Formula: Loss = -alpha * (1 - p_t)^gamma * log(p_t)

    Attributes:
        alpha (float): Weighting factor for the rare class (positive class).
        gamma (float): Focusing parameter to down-weight easy examples.
        reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Logits from the model (N, 1) or (N,).
            targets (torch.Tensor): Ground truth labels (N, 1) or (N,).

        Returns:
            torch.Tensor: Computed Focal Loss.
        """
        # Ensure inputs and targets are float32 for stability in AMP
        inputs = inputs.float()
        targets = targets.float()

        # Reshape if necessary to match
        if inputs.shape != targets.shape:
            targets = targets.view_as(inputs)

        # Compute binary cross entropy loss (which is -log(p_t))
        # reduction='none' is essential to apply the focal weights element-wise
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

        # Calculate p_t (probability of the true class)
        # p_t = p if y=1 else 1-p
        # We can derive p_t from bce_loss: bce_loss = -log(p_t) => p_t = exp(-bce_loss)
        p_t = torch.exp(-bce_loss)

        # Calculate the focal term: (1 - p_t)^gamma
        focal_term = (1 - p_t) ** self.gamma

        # Calculate alpha weighting
        # alpha_t = alpha if y=1 else (1-alpha)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Combine terms
        loss = alpha_t * focal_term * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class MultiTaskLoss(nn.Module):
    """
    Composite loss function for the CMT-SIN model.
    Combines the primary Focal Loss for cancer detection with CrossEntropy losses
    for auxiliary tasks (BIRADS, Density).
    """

    def __init__(self):
        super(MultiTaskLoss, self).__init__()

        # Primary Task Loss
        self.cancer_loss_fn = FocalLoss(
            alpha=Config.FOCAL_ALPHA, gamma=Config.FOCAL_GAMMA, reduction="mean"
        )

        # Auxiliary Task Losses
        # We use ignore_index=-100 to handle missing labels (NaNs in metadata are mapped to -100 in Dataset)
        self.aux_loss_fns = nn.ModuleDict()
        self.aux_weights = {}

        if Config.USE_AUX_TASKS:
            for task_name, task_config in Config.AUX_TASKS.items():
                self.aux_loss_fns[task_name] = nn.CrossEntropyLoss(ignore_index=-100)
                self.aux_weights[task_name] = task_config["loss_weight"]

    def forward(self, preds, targets):
        """
        Args:
            preds (dict): Dictionary containing model outputs.
                          Keys: 'cancer', 'BIRADS', 'density'.
            targets (dict): Dictionary containing ground truth.
                          Keys: 'cancer', 'BIRADS', 'density'.

        Returns:
            dict: Dictionary containing 'total_loss' and individual loss components.
        """
        losses = {}

        # 1. Primary Cancer Loss
        # Ensure we have the logits for cancer
        if "cancer" in preds and "cancer" in targets:
            cancer_loss = self.cancer_loss_fn(preds["cancer"], targets["cancer"])
            losses["cancer_loss"] = cancer_loss
            total_loss = cancer_loss
        else:
            # Should not happen in normal training flow
            total_loss = torch.tensor(0.0, device=Config.DEVICE)
            losses["cancer_loss"] = total_loss

        # 2. Auxiliary Losses
        if Config.USE_AUX_TASKS:
            for task_name, loss_fn in self.aux_loss_fns.items():
                if task_name in preds and task_name in targets:
                    pred_logits = preds[task_name]
                    target_labels = targets[task_name]

                    # Compute loss (CrossEntropyLoss handles ignore_index internally)
                    # We cast targets to long as they are class indices
                    task_loss = loss_fn(pred_logits, target_labels.long())

                    # Check if loss is NaN (can happen if all targets in batch are -100)
                    if torch.isnan(task_loss):
                        task_loss = torch.tensor(0.0, device=Config.DEVICE)

                    weight = self.aux_weights.get(task_name, 1.0)
                    losses[f"{task_name}_loss"] = task_loss

                    total_loss = total_loss + (weight * task_loss)

        losses["total_loss"] = total_loss
        return losses
