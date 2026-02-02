import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config

# =================================================================================
# Lovasz-Softmax / Lovasz-Hinge Helper Functions
# Source: https://github.com/bermanmaxim/LovaszSoftmax
# =================================================================================


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
    errors = 1.0 - logits * torch.autograd.Variable(signs)
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), torch.autograd.Variable(grad))
    return loss


class LovaszHingeLoss(nn.Module):
    """
    Wrapper for Lovasz Hinge Loss.
    """

    def __init__(self):
        super().__init__()

    def forward(self, logits, targets):
        """
        Args:
            logits: (N, 1, H, W) or (N, H, W)
            targets: (N, 1, H, W) or (N, H, W)
        """
        # Flatten
        logits_flat = logits.view(-1)
        targets_flat = targets.view(-1)
        return lovasz_hinge_flat(logits_flat, targets_flat)


# =================================================================================
# Teacher Loss (Supervised)
# =================================================================================


class TeacherLoss(nn.Module):
    """
    Loss function for the Specialist Teacher.
    Combines BCEWithLogitsLoss and LovaszHingeLoss.
    """

    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.lovasz = LovaszHingeLoss()

    def forward(self, logits, targets):
        """
        Args:
            logits: Mask logits (N, 1, H, W)
            targets: Binary Ground Truth Masks (N, 1, H, W)
        """
        bce_loss = self.bce(logits, targets)
        lovasz_loss = self.lovasz(logits, targets)
        return bce_loss + lovasz_loss


# =================================================================================
# Student Loss (Multi-Task & Distillation)
# =================================================================================


class StudentLoss(nn.Module):
    """
    Loss function for the Generalist Student.
    Handles two modes:
    1. Labeled (Supervised): BCE + Lovasz (Mask) + MSE (Depth)
    2. Unlabeled (Distillation): BCE (Mask vs Soft Targets)
    """

    def __init__(self):
        super().__init__()
        # BCE for both hard and soft targets
        self.bce = nn.BCEWithLogitsLoss()
        self.lovasz = LovaszHingeLoss()
        self.mse = nn.MSELoss()

    def forward(self, mask_logits, depth_preds, mask_targets, depth_targets=None):
        """
        Args:
            mask_logits: (N, 1, H, W)
            depth_preds: (N, 1) - Predicted depth scalars
            mask_targets: (N, 1, H, W) - Can be binary (GT) or float (Soft Pseudo)
            depth_targets: (N, 1) or None - GT depth. If None, assumes Unlabeled/Distillation mode.
        """
        # Distillation Mode (Unlabeled / Soft Targets)
        # We use BCE only because Lovasz is incompatible with soft targets.
        # We do not use MSE because we don't have ground truth depth.
        if depth_targets is None:
            return self.bce(mask_logits, mask_targets)

        # Supervised Mode (Labeled)
        # Mask Loss: BCE + Lovasz
        mask_loss = self.bce(mask_logits, mask_targets) + self.lovasz(
            mask_logits, mask_targets
        )

        # Depth Loss: MSE
        # Ensure depth targets are float32
        depth_loss = self.mse(depth_preds, depth_targets.float())

        return mask_loss + depth_loss
