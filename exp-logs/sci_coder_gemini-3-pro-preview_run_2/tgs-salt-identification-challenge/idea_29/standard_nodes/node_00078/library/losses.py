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
