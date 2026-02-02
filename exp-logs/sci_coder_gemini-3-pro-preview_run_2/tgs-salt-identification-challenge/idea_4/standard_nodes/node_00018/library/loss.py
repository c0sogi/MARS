import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

try:
    from itertools import ifilterfalse
except ImportError:
    from itertools import filterfalse as ifilterfalse


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Jaccard loss w.r.t the sorted ground truth.
    Args:
        gt_sorted: Sorted ground truth labels (1 for salt, 0 for sediment).
    Returns:
        Gradient of the Jaccard loss.
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:  # Cover case where p=1 to avoid error
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    r"""
    Binary Lovasz hinge loss.
    Args:
        logits: [P] Logits at each pixel (between -\infty and +\infty)
        labels: [P] Binary ground truth labels (0 or 1)
        per_image: Compute the loss per image instead of per batch
        ignore: Void class id
    Returns:
        The calculated loss.
    """
    if per_image:
        loss = mean(
            lovasz_hinge_flat(
                *flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore)
            )
            for log, lab in zip(logits, labels)
        )
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss


def lovasz_hinge_flat(logits, labels):
    r"""
    Binary Lovasz hinge loss on flattened tensors.
    Args:
        logits: [P] Logits at each pixel (between -\infty and +\infty)
        labels: [P] Binary ground truth labels (0 or 1)
    Returns:
        The calculated loss.
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.0

    signs = 2.0 * labels.float() - 1.0
    errors = 1.0 - logits * Variable(signs)
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), Variable(grad))
    return loss


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case).
    Remove labels equal to 'ignore'.
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = labels != ignore
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels


def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(torch.isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == "raise":
            raise ValueError("Empty mean")
        return empty
    for x in l:
        n += 1
        acc += x
    return acc / n


class LovaszHingeLoss(nn.Module):
    """
    Wrapper for Lovasz Hinge Loss.
    """

    def __init__(self, per_image=True, ignore=None):
        super().__init__()
        self.per_image = per_image
        self.ignore = ignore

    def forward(self, logits, targets):
        """
        Args:
            logits: (N, C, H, W) or (N, H, W) logits (no sigmoid applied).
            targets: (N, H, W) or (N, 1, H, W) binary masks (0 or 1).
        """
        return lovasz_hinge(
            logits, targets, per_image=self.per_image, ignore=self.ignore
        )


class BCELovaszLoss(nn.Module):
    """
    Composite loss function: BCEWithLogitsLoss + LovaszHingeLoss.
    """

    def __init__(self, bce_weight=0.5, lovasz_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.lovasz_weight = lovasz_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.lovasz = LovaszHingeLoss()

    def forward(self, logits, targets):
        """
        Args:
            logits: (N, 1, H, W) or (N, H, W) logits.
            targets: (N, 1, H, W) or (N, H, W) binary masks.
        """
        # Ensure targets are float for BCE
        targets_float = targets.float()

        # BCE expects same shape, Lovasz handles flattening internally
        # If logits are (N, 1, H, W) and targets are (N, H, W), unsqueeze targets for BCE
        if logits.dim() == 4 and targets_float.dim() == 3:
            targets_float = targets_float.unsqueeze(1)

        # Calculate BCE
        bce_loss = self.bce(logits, targets_float)

        # Calculate Lovasz
        # Lovasz implementation handles flattening, but we ensure basic shape compatibility
        lovasz_loss = self.lovasz(logits, targets)

        return (self.bce_weight * bce_loss) + (self.lovasz_weight * lovasz_loss)
