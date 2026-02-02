import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Jaccard loss w.r.t the sorted error
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
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
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
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
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        loss = 0
        for input_i, target_i in zip(logits, labels):
            loss = loss + lovasz_hinge_flat(
                *flatten_binary_scores(input_i, target_i, ignore)
            )
        return loss / logits.size(0)
    else:
        return lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))


class LovaszHingeLoss(nn.Module):
    """
    Wrapper for Lovasz Hinge Loss.
    """

    def __init__(self, per_image=True, ignore=None):
        super().__init__()
        self.per_image = per_image
        self.ignore = ignore

    def forward(self, logits, labels):
        return lovasz_hinge(
            logits, labels, per_image=self.per_image, ignore=self.ignore
        )


class SegmentationLoss(nn.Module):
    """
    Combined Segmentation Loss (Lovasz + BCE).
    """

    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.lovasz = LovaszHingeLoss()

    def forward(self, mask_logits, gt_mask):
        gt_mask = gt_mask.float()

        # Squeeze channel dim for Lovasz if present (B, 1, H, W) -> (B, H, W)
        if mask_logits.dim() == 4 and mask_logits.shape[1] == 1:
            mask_logits_flat = mask_logits.squeeze(1)
        else:
            mask_logits_flat = mask_logits

        if gt_mask.dim() == 4 and gt_mask.shape[1] == 1:
            gt_mask_flat = gt_mask.squeeze(1)
        else:
            gt_mask_flat = gt_mask

        bce_loss = self.bce(mask_logits, gt_mask)
        lovasz_loss = self.lovasz(mask_logits_flat, gt_mask_flat)

        # Summing losses (1.0 + 1.0)
        total_loss = bce_loss + lovasz_loss

        return total_loss, {
            "loss": total_loss.item(),
            "bce": bce_loss.item(),
            "lovasz": lovasz_loss.item(),
        }
