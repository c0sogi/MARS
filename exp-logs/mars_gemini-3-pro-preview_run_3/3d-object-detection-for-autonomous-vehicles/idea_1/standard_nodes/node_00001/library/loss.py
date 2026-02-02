import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class SigmoidFocalLoss(nn.Module):
    """
    Sigmoid Focal Loss for handling class imbalance.
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets, weights=None):
        """
        Args:
            inputs: (N, ) tensor of logits.
            targets: (N, ) tensor of binary targets (0 or 1).
            weights: (N, ) tensor of sample weights (optional).
        """
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if weights is not None:
            loss = loss * weights

        return loss.sum()


class WeightedSmoothL1Loss(nn.Module):
    """
    Smooth L1 Loss for regression tasks.
    """

    def __init__(self, beta: float = 1.0 / 9.0):
        super().__init__()
        self.beta = beta

    def forward(self, inputs, targets, weights=None):
        """
        Args:
            inputs: (N, 7) tensor of predicted offsets.
            targets: (N, 7) tensor of target offsets.
            weights: (N, ) tensor of weights (optional).
        """
        loss = F.smooth_l1_loss(inputs, targets, reduction="none", beta=self.beta)

        if weights is not None:
            loss = loss * weights.unsqueeze(-1)

        return loss.sum()


class SoftmaxCrossEntropyLoss(nn.Module):
    """
    Softmax Cross Entropy Loss for direction classification (optional).
    """

    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets):
        return F.cross_entropy(inputs, targets, reduction="mean")


class MultiTaskLoss(nn.Module):
    """
    Combines Classification and Regression losses for PointPillars.
    """

    def __init__(self):
        super().__init__()
        self.cls_loss_func = SigmoidFocalLoss(
            alpha=Config.FOCAL_ALPHA, gamma=Config.FOCAL_GAMMA
        )
        self.reg_loss_func = WeightedSmoothL1Loss(beta=1.0 / 9.0)

        self.cls_weight = Config.LOSS_WEIGHTS.get("cls_weight", 1.0)
        self.loc_weight = Config.LOSS_WEIGHTS.get("loc_weight", 2.0)
        self.dir_weight = Config.LOSS_WEIGHTS.get("dir_weight", 0.2)

    def forward(self, cls_preds, reg_preds, cls_targets, reg_targets):
        """
        Calculate the total weighted loss.

        Args:
            cls_preds: (B, Num_Anchors_Per_Loc, H, W)
            reg_preds: (B, Num_Anchors_Per_Loc * 7, H, W)
            cls_targets: (B, Total_Anchors) - Flattened targets
            reg_targets: (B, Total_Anchors, 7) - Flattened regression targets

        Returns:
            dict: Dictionary containing 'loss', 'cls_loss', 'loc_loss'.
        """
        # 1. Flatten Predictions to match Targets
        # (B, C, H, W) -> (B, H, W, C) -> (-1, C_per_anchor) or (-1)

        # cls_preds: (B, 18, H, W) -> (B, H, W, 18) -> (-1)
        # Note: The model output is one score per anchor (binary classification for that anchor)
        batch_size = cls_preds.shape[0]
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous().view(-1)

        # reg_preds: (B, 18*7, H, W) -> (B, H, W, 18, 7) -> (-1, 7)
        reg_preds = reg_preds.permute(0, 2, 3, 1).contiguous().view(-1, 7)

        # Ensure targets are flattened
        cls_targets = cls_targets.view(-1)
        reg_targets = reg_targets.view(-1, 7)

        # 2. Generate Masks
        # cls_targets: 0 = Background, -1 = Ignore, >0 = Class ID
        pos_mask = cls_targets > 0
        valid_mask = cls_targets != -1

        num_pos = pos_mask.sum().float()
        num_pos = torch.clamp(num_pos, min=1.0)  # Avoid division by zero

        # 3. Classification Loss (Focal Loss)
        # We treat this as a binary classification problem for each anchor:
        # Does this anchor contain its assigned class?
        # Targets for Focal Loss: 1 if positive, 0 if background. Ignored are masked out later.
        labels = torch.zeros_like(cls_preds)
        labels[pos_mask] = 1.0

        # Calculate loss on all valid anchors (pos + neg, excluding ignore)
        # We compute on all, then mask.
        # However, SigmoidFocalLoss expects inputs and targets.
        # We pass full tensors and mask the result or pass masked tensors.
        # Passing masked tensors is safer for reduction.

        cls_loss = self.cls_loss_func(cls_preds[valid_mask], labels[valid_mask])
        cls_loss = cls_loss / num_pos

        # 4. Regression Loss (Smooth L1)
        # Only calculated on positive anchors
        loc_loss = torch.tensor(0.0, device=cls_preds.device)
        if pos_mask.any():
            loc_loss = self.reg_loss_func(reg_preds[pos_mask], reg_targets[pos_mask])
            loc_loss = loc_loss / num_pos

        # 5. Total Loss
        total_loss = self.cls_weight * cls_loss + self.loc_weight * loc_loss

        return {"loss": total_loss, "cls_loss": cls_loss, "loc_loss": loc_loss}


def compute_loss(cls_preds, reg_preds, cls_targets, reg_targets):
    """
    Helper function to instantiate and run the loss calculation.
    """
    criterion = MultiTaskLoss()
    return criterion(cls_preds, reg_preds, cls_targets, reg_targets)
