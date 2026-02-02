import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ============================================================================
# Lovasz-Softmax / Lovasz-Hinge Implementation
# Reference: https://github.com/bermanmaxim/LovaszSoftmax
# ============================================================================


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.cumsum(0)
    union = gts + (1 - gt_sorted).cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each pixel (between -\\infty and +\\infty)
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


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = labels != ignore
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels


class LovaszHingeLoss(nn.Module):
    """
    Binary Lovasz Hinge Loss for semantic segmentation.
    This loss directly optimizes the Jaccard index (IoU).
    """

    def __init__(self, ignore_index=None):
        super(LovaszHingeLoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, 1, H, W) or (B, H, W) logits
            targets: (B, 1, H, W) or (B, H, W) binary targets (0 or 1)
        """
        # Cite solution_lesson_node_00044: Prefer Per-Image Loss Aggregation
        losses = []
        for i in range(logits.size(0)):
            single_logit = logits[i]
            single_target = targets[i]
            loss = lovasz_hinge_flat(
                *flatten_binary_scores(
                    single_logit, single_target, ignore=self.ignore_index
                )
            )
            losses.append(loss)
        return torch.stack(losses).mean()


# ============================================================================
# Composite Losses
# ============================================================================


class MixedLoss(nn.Module):
    """
    Combines BCEWithLogitsLoss and LovaszHingeLoss.
    Used for the Teacher model and the labeled part of the Student model.
    """

    def __init__(self, bce_weight=1.0, lovasz_weight=1.0):
        super(MixedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.lovasz_weight = lovasz_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.lovasz = LovaszHingeLoss()

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, 1, H, W)
            targets: (B, 1, H, W)
        """
        loss = 0.0
        if self.bce_weight > 0:
            loss += self.bce_weight * self.bce(logits, targets)

        if self.lovasz_weight > 0:
            loss += self.lovasz_weight * self.lovasz(logits, targets)

        return loss


class StudentLoss(nn.Module):
    """
    Multi-task loss for the Generalist Student.

    Handles two modes:
    1. Labeled Data:
       - Mask Loss: BCE + Lovasz (against hard GT)
       - Depth Loss: MSE (against true depth)

    2. Unlabeled/Soft Data:
       - Mask Loss: BCE (against soft pseudo-labels)
       - Depth Loss: Ignored (or 0)

    The mode is determined by the presence of `depth_targets`.
    """

    def __init__(self, bce_weight=1.0, lovasz_weight=1.0, depth_weight=1.0):
        super(StudentLoss, self).__init__()
        self.bce_weight = bce_weight
        self.lovasz_weight = lovasz_weight
        self.depth_weight = depth_weight

        # Components
        self.bce = nn.BCEWithLogitsLoss()
        self.lovasz = LovaszHingeLoss()
        self.mse = nn.MSELoss()

    def forward(self, mask_logits, depth_pred, mask_targets, depth_targets=None):
        """
        Args:
            mask_logits: (B, 1, H, W) predicted mask logits.
            depth_pred: (B, 1) predicted depth scalars.
            mask_targets: (B, 1, H, W) target masks.
                          Can be binary (0/1) for labeled data,
                          or soft probabilities (0.0-1.0) for unlabeled data.
            depth_targets: (B, 1) or None. True depth values.
                           If None, assumes unlabeled/soft-target mode.

        Returns:
            total_loss
        """
        # Case 1: Labeled Data (Hard targets + Depth available)
        if depth_targets is not None:
            # Mask Loss: Mixed BCE + Lovasz
            seg_loss = 0.0
            if self.bce_weight > 0:
                seg_loss += self.bce_weight * self.bce(mask_logits, mask_targets)
            if self.lovasz_weight > 0:
                seg_loss += self.lovasz_weight * self.lovasz(mask_logits, mask_targets)

            # Depth Loss: MSE
            # depth_pred and depth_targets should be (B, 1)
            reg_loss = self.mse(depth_pred, depth_targets)

            total_loss = seg_loss + (self.depth_weight * reg_loss)
            return total_loss

        # Case 2: Unlabeled Data (Soft targets, Depth unknown/marginalized)
        else:
            # Mask Loss: BCE only (Lovasz is not defined for soft targets)
            # BCEWithLogitsLoss handles soft targets (probabilities) correctly by design
            # (it computes cross entropy between prob distribution and logits)
            seg_loss = self.bce(mask_logits, mask_targets)

            return seg_loss
