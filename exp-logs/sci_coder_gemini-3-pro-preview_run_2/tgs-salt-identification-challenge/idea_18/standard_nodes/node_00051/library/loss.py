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
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
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
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth labels (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        loss = 0
        for input, target in zip(logits, labels):
            input = input.view(-1)
            target = target.view(-1)
            if ignore is not None:
                valid = target != ignore
                input = input[valid]
                target = target[valid]
            loss = loss + lovasz_hinge_flat(input, target)
        return loss / logits.size(0)
    else:
        logits = logits.view(-1)
        labels = labels.view(-1)
        if ignore is not None:
            valid = labels != ignore
            logits = logits[valid]
            labels = labels[valid]
        return lovasz_hinge_flat(logits, labels)


class LovaszHingeLoss(nn.Module):
    """
    Wrapper for Lovasz Hinge Loss.
    Directly optimizes the Jaccard Index (IoU).
    """

    def __init__(self, per_image=True, ignore=None):
        super().__init__()
        self.per_image = per_image
        self.ignore = ignore

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, 1, H, W) or (B, H, W)
            targets: (B, 1, H, W) or (B, H, W)
        """
        # Squeeze channel dimension if present
        if logits.dim() > 3:
            logits = logits.squeeze(1)
        if targets.dim() > 3:
            targets = targets.squeeze(1)

        return lovasz_hinge(
            logits, targets, per_image=self.per_image, ignore=self.ignore
        )


class BCELovaszLoss(nn.Module):
    """
    Composite loss combining Binary Cross Entropy and Lovasz Hinge Loss.
    BCE provides smooth gradients for pixel-wise accuracy.
    Lovasz optimizes the global IoU metric.
    """

    def __init__(self, bce_weight=1.0, lovasz_weight=1.0, per_image=True):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.lovasz = LovaszHingeLoss(per_image=per_image)
        self.bce_weight = bce_weight
        self.lovasz_weight = lovasz_weight

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, 1, H, W) raw scores
            targets: (B, 1, H, W) binary masks (0 or 1)
        """
        # BCE expects Float targets, same shape as logits
        bce_loss = self.bce(logits, targets.float())

        # Lovasz handles squeezing internally, but expects binary targets
        lovasz_loss = self.lovasz(logits, targets)

        return (self.bce_weight * bce_loss) + (self.lovasz_weight * lovasz_loss)
