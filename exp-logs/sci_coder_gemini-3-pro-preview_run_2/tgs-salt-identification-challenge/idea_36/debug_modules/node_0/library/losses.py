import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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
            input_flat, target_flat = flatten_binary_scores(input, target, ignore)
            loss += lovasz_hinge_flat(input_flat, target_flat)
        return loss / logits.size(0)
    else:
        input_flat, target_flat = flatten_binary_scores(logits, labels, ignore)
        return lovasz_hinge_flat(input_flat, target_flat)


class MixedLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, input, target):
        """
        Args:
            input: (B, 1, H, W) or (B, H, W) logits
            target: (B, 1, H, W) or (B, H, W) binary mask (0 or 1)
        """
        # Ensure target is float for BCE
        target_float = target.float()

        # Calculate BCE
        loss_bce = self.bce(input, target_float)

        # Calculate Lovasz
        # Lovasz implementation handles flattening internally, but expects matching dims
        # or it iterates over batch.
        loss_lovasz = lovasz_hinge(input, target_float)

        return self.alpha * loss_bce + self.beta * loss_lovasz
