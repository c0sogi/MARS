import torch
import torch.nn as nn
import torch.nn.functional as F


class LovaszHingeLoss(nn.Module):
    """
    Lovasz-Hinge Loss for binary segmentation.
    Optimizes the Jaccard index (IoU) directly using the Lovasz extension.
    """

    def __init__(self, per_image=True):
        super(LovaszHingeLoss, self).__init__()
        self.per_image = per_image

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): Logits of shape (B, 1, H, W) or (B, H, W).
            targets (torch.Tensor): Binary mask of shape (B, 1, H, W) or (B, H, W).

        Returns:
            torch.Tensor: Scalar loss.
        """
        # Squeeze channel dimension if present
        if logits.dim() == 4 and logits.size(1) == 1:
            logits = logits.squeeze(1)
        if targets.dim() == 4 and targets.size(1) == 1:
            targets = targets.squeeze(1)

        if self.per_image:
            loss = 0
            batch_size = logits.size(0)
            for i in range(batch_size):
                loss += self.lovasz_hinge_flat(logits[i], targets[i])
            return loss / batch_size
        else:
            return self.lovasz_hinge_flat(logits, targets)

    def lovasz_hinge_flat(self, logits, labels):
        """
        Binary Lovasz hinge loss for a flat vector.

        Args:
            logits: (N,) logits.
            labels: (N,) binary targets (0 or 1).
        """
        if len(logits) == 0:
            return logits.sum() * 0.0

        logits = logits.view(-1)
        labels = labels.view(-1)

        # Signs: -1 for class 0, +1 for class 1
        signs = 2.0 * labels.float() - 1.0

        # Hinge errors: relu(1 - logits * signs)
        errors = 1.0 - logits * signs

        # Sort errors descending
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        perm = perm.data

        # Sort targets by error permutation
        gt_sorted = labels.float()[perm]

        # Compute gradient of Lovasz extension
        grad = self.lovasz_grad(gt_sorted)

        # Loss is dot product of sorted errors (relu'd) and gradient
        loss = torch.dot(F.relu(errors_sorted), grad)
        return loss

    def lovasz_grad(self, gt_sorted):
        """
        Computes gradient of the Jaccard loss w.r.t the errors.
        """
        p = len(gt_sorted)
        gts = gt_sorted.sum()

        # Intersection and Union calculation
        intersection = gts - gt_sorted.cumsum(0)
        union = gts + (1 - gt_sorted).cumsum(0)

        # Jaccard Index
        jaccard = 1.0 - intersection / union

        # The gradient is the difference in Jaccard index
        if p > 1:
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]

        return jaccard


class StableBCELoss(nn.Module):
    """
    Stable Binary Cross Entropy Loss using BCEWithLogitsLoss.
    Handles both binary and soft targets.
    """

    def __init__(self):
        super(StableBCELoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): Logits.
            targets (torch.Tensor): Targets (binary or soft probabilities).
        """
        return self.bce(logits, targets.float())


class MSELoss(nn.Module):
    """
    Mean Squared Error Loss for auxiliary depth regression.
    """

    def __init__(self):
        super(MSELoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, preds, targets):
        """
        Args:
            preds (torch.Tensor): Predicted values.
            targets (torch.Tensor): Target values.
        """
        return self.mse(preds, targets)


def calc_combined_loss(
    pred_mask, target_mask, pred_depth=None, target_depth=None, soft_targets=False
):
    """
    Aggregates losses based on the training stage.

    Logic:
        - Base: StableBCELoss (Mask)
        - If Hard Targets (not soft_targets): Add LovaszHingeLoss (Mask)
        - If Depth provided (pred_depth & target_depth): Add MSELoss (Depth)

    Args:
        pred_mask (torch.Tensor): Predicted mask logits (B, 1, H, W) or (B, H, W).
        target_mask (torch.Tensor): Ground truth mask (B, 1, H, W) or (B, H, W).
                                    If soft_targets=True, these are probabilities [0,1].
        pred_depth (torch.Tensor, optional): Predicted depth scalars (B, 1).
        target_depth (torch.Tensor, optional): Ground truth depth scalars (B, 1).
        soft_targets (bool): Flag indicating if targets are soft probabilities (Distillation).

    Returns:
        torch.Tensor: The combined scalar loss.
    """
    # Ensure inputs are FP32 for stability
    pred_mask = pred_mask.float()
    target_mask = target_mask.float()

    # 1. BCE Loss (Always applied)
    # Handles binary classification and soft-target distillation
    bce_func = nn.BCEWithLogitsLoss()
    total_loss = bce_func(pred_mask, target_mask)

    # 2. Lovasz Hinge Loss (Only for Hard Targets)
    # Lovasz is designed for binary ground truth and optimizes IoU directly
    if not soft_targets:
        lovasz_loss_func = LovaszHingeLoss(per_image=True)
        total_loss += lovasz_loss_func(pred_mask, target_mask)

    # 3. Auxiliary Depth Loss (If depth heads are active)
    if pred_depth is not None and target_depth is not None:
        pred_depth = pred_depth.float()
        target_depth = target_depth.float()

        # Ensure shapes match
        if pred_depth.shape != target_depth.shape:
            target_depth = target_depth.view_as(pred_depth)

        mse_func = nn.MSELoss()
        total_loss += mse_func(pred_depth, target_depth)

    return total_loss
