import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.0

    signs = 2.0 * labels.float() - 1.0
    errors = 1.0 - logits * signs
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), grad)
    return loss


class LovaszHingeLoss(nn.Module):
    """
    Lovasz Hinge Loss for Binary Segmentation.
    Optimizes the Jaccard index (IoU) directly.
    """

    def __init__(self):
        super(LovaszHingeLoss, self).__init__()

    def forward(self, logits, labels):
        """
        Args:
            logits: (N, 1, H, W) or (N, H, W) logits
            labels: (N, 1, H, W) or (N, H, W) binary labels (0 or 1)
        """
        # Flatten the tensors
        logits_flat = logits.view(-1)
        labels_flat = labels.view(-1)

        return lovasz_hinge_flat(logits_flat, labels_flat)


class MultiTaskLoss(nn.Module):
    """
    Combined loss for Multi-Task Wide-LinkNet.

    Components:
    1. Segmentation: Lovasz Hinge Loss + BCEWithLogitsLoss
    2. Depth Regression: MSELoss (weighted by aux_weight)
    """

    def __init__(self, aux_weight=Config.AUX_DEPTH_LOSS_WEIGHT):
        super(MultiTaskLoss, self).__init__()
        self.aux_weight = aux_weight

        # Segmentation losses
        self.bce = nn.BCEWithLogitsLoss()
        self.lovasz = LovaszHingeLoss()

        # Depth loss
        self.mse = nn.MSELoss()

    def forward(self, pred_mask, pred_depth, true_mask, true_depth=None):
        """
        Args:
            pred_mask (torch.Tensor): Predicted segmentation logits (N, 1, H, W)
            pred_depth (torch.Tensor): Predicted depth scalars (N, 1)
            true_mask (torch.Tensor): Ground truth segmentation mask (N, 1, H, W)
            true_depth (torch.Tensor, optional): Ground truth depth scalars (N, 1) or (N,)

        Returns:
            torch.Tensor: Weighted sum of losses
            dict: Dictionary containing individual loss components for logging
        """
        # 1. Segmentation Loss
        # BCE
        loss_bce = self.bce(pred_mask, true_mask)

        # Lovasz
        loss_lovasz = self.lovasz(pred_mask, true_mask)

        loss_seg = loss_bce + loss_lovasz

        # 2. Depth Loss (Auxiliary)
        loss_depth = torch.tensor(0.0, device=pred_mask.device)

        if pred_depth is not None and true_depth is not None and self.aux_weight > 0:
            # Ensure shapes match for MSE
            if true_depth.dim() == 1:
                true_depth = true_depth.view(-1, 1)

            # Ensure pred_depth is the right shape
            if pred_depth.dim() == 1:
                pred_depth = pred_depth.view(-1, 1)

            loss_depth = self.mse(pred_depth, true_depth)

        # Total Loss
        total_loss = loss_seg + (self.aux_weight * loss_depth)

        metrics = {
            "loss_bce": loss_bce.item(),
            "loss_lovasz": loss_lovasz.item(),
            "loss_depth": loss_depth.item(),
            "loss_total": total_loss.item(),
        }

        return total_loss, metrics
