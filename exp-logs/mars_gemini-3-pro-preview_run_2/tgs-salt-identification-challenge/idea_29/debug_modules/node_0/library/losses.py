import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config

# -------------------------------------------------------------------------
# Lovasz-Hinge Loss Implementation
# -------------------------------------------------------------------------


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Jaccard loss w.r.t the sorted error.
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
    Binary Lovasz hinge loss on flattened tensors.
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.0

    signs = 2.0 * labels.float() - 1.0
    errors = 1.0 - logits * torch.autograd.Variable(signs)
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), torch.autograd.Variable(grad))
    return loss


class LovaszHingeLoss(nn.Module):
    """
    Lovasz Hinge Loss for Binary Segmentation.
    Optimizes the Jaccard index directly.
    """

    def __init__(self):
        super().__init__()

    def forward(self, logits, masks):
        """
        Args:
            logits: (N, 1, H, W) or (N, H, W) logits from the model.
            masks: (N, 1, H, W) or (N, H, W) binary ground truth masks (0 or 1).
        """
        # Flatten
        logits_flat = logits.view(-1)
        masks_flat = masks.view(-1)

        loss = lovasz_hinge_flat(logits_flat, masks_flat)
        return loss


# -------------------------------------------------------------------------
# Multi-Task Student Loss Implementation
# -------------------------------------------------------------------------


class StudentLoss(nn.Module):
    """
    Composite loss function for the Multi-Task Student model.
    Combines:
    1. Segmentation Loss: Lovasz Hinge + BCE
    2. Depth Regression Loss: MSE (Auxiliary Task)
    3. Distillation Loss: BCE (Student Logits vs Teacher Soft Targets)
    """

    def __init__(self):
        super().__init__()
        self.lovasz = LovaszHingeLoss()
        self.bce_seg = nn.BCEWithLogitsLoss()
        self.mse_depth = nn.MSELoss()
        self.bce_distill = nn.BCEWithLogitsLoss()

        # Weights from Config
        self.w_seg = Config.LOSS_WEIGHT_SEG
        self.w_mse = Config.LOSS_WEIGHT_MSE
        self.w_distill = Config.LOSS_WEIGHT_BCE_DISTILL

    def forward(self, student_outputs, masks, depths, teacher_logits=None):
        """
        Args:
            student_outputs (dict): Dictionary containing:
                - 'logits': (N, 1, H, W) Segmentation logits
                - 'depth': (N, 1) Predicted depth scalars
            masks (torch.Tensor): (N, 1, H, W) Ground truth binary masks.
            depths (torch.Tensor): (N, 1) Ground truth depth values (normalized).
            teacher_logits (torch.Tensor, optional): (N, 1, H, W) Logits from the teacher model.
                                                     If None, distillation loss is skipped.

        Returns:
            total_loss (torch.Tensor): Weighted sum of all losses.
            metrics (dict): Dictionary of individual loss components for logging.
        """
        pred_logits = student_outputs["logits"]
        pred_depth = student_outputs["depth"]

        # 1. Segmentation Loss (Lovasz + BCE)
        # Ensure masks are float for BCE and Lovasz
        masks = masks.float()

        loss_lovasz = self.lovasz(pred_logits, masks)
        loss_bce_seg = self.bce_seg(pred_logits, masks)
        loss_seg = loss_lovasz + loss_bce_seg

        # 2. Depth Regression Loss (MSE)
        # Ensure depths are float
        depths = depths.float().view_as(pred_depth)
        loss_depth = self.mse_depth(pred_depth, depths)

        # 3. Distillation Loss (BCE with Soft Targets)
        loss_distill = torch.tensor(0.0, device=pred_logits.device)
        if teacher_logits is not None:
            # Teacher logits -> Probabilities (Soft Targets)
            teacher_probs = torch.sigmoid(teacher_logits)
            # BCEWithLogitsLoss accepts soft targets (probabilities) as the target argument
            loss_distill = self.bce_distill(pred_logits, teacher_probs)

        # Combine Losses
        # L = L_Seg + 0.1 * L_MSE + 0.5 * L_Distill
        total_loss = (
            (self.w_seg * loss_seg)
            + (self.w_mse * loss_depth)
            + (self.w_distill * loss_distill)
        )

        # Metrics for logging
        metrics = {
            "loss_total": total_loss.item(),
            "loss_seg": loss_seg.item(),
            "loss_lovasz": loss_lovasz.item(),
            "loss_bce_seg": loss_bce_seg.item(),
            "loss_depth": loss_depth.item(),
            "loss_distill": loss_distill.item(),
        }

        return total_loss, metrics
