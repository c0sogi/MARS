import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Jaccard loss w.r.t the sorted error
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
    Binary Lovasz hinge loss on flat values
      logits: [P] Tensor, logits at each pixel (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth masks (0 or 1)
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


def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Tensor, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        loss = 0
        # If logits has channel dim 1, squeeze it for iteration
        if logits.dim() == 4 and logits.shape[1] == 1:
            logits_iter = logits.squeeze(1)
        else:
            logits_iter = logits

        # If labels has channel dim 1, squeeze it
        if labels.dim() == 4 and labels.shape[1] == 1:
            labels_iter = labels.squeeze(1)
        else:
            labels_iter = labels

        # Ensure batch size matches
        if logits_iter.size(0) != labels_iter.size(0):
            raise ValueError(
                f"Batch size mismatch: logits {logits.shape}, labels {labels.shape}"
            )

        for input, target in zip(logits_iter, labels_iter):
            loss = loss + lovasz_hinge_flat(input.flatten(), target.flatten())
        return loss / logits.size(0)
    else:
        return lovasz_hinge_flat(logits.flatten(), labels.flatten())


class LovaszHingeLoss(nn.Module):
    """
    Wrapper for Lovasz Hinge Loss to be used as a PyTorch Module.
    """

    def __init__(self, per_image=True):
        super().__init__()
        self.per_image = per_image

    def forward(self, logits, targets):
        return lovasz_hinge(logits, targets, per_image=self.per_image)


class CombinedLoss(nn.Module):
    """
    Combines Binary Cross Entropy (BCE) and Lovasz Hinge Loss.
    Useful for stabilizing training while optimizing for IoU.
    """

    def __init__(self, bce_weight=0.5, lovasz_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.lovasz_weight = lovasz_weight
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.lovasz_loss = LovaszHingeLoss()

    def forward(self, logits, targets):
        # Prepare inputs for BCE (requires float targets and matching shapes)
        logits_bce = logits
        targets_bce = targets

        # Squeeze channel dimension if present (B, 1, H, W) -> (B, H, W)
        if logits.dim() == 4 and logits.shape[1] == 1:
            logits_bce = logits.squeeze(1)

        if targets.dim() == 4 and targets.shape[1] == 1:
            targets_bce = targets.squeeze(1)

        # Ensure targets match logits shape for BCE
        if logits_bce.shape != targets_bce.shape:
            # Fallback: try to view targets as logits
            targets_bce = targets_bce.view_as(logits_bce)

        bce = self.bce_loss(logits_bce, targets_bce.float())
        lovasz = self.lovasz_loss(logits, targets)

        return self.bce_weight * bce + self.lovasz_weight * lovasz
