import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Jaccard extension w.r.t the sorted errors
    See Alg. 1 in https://arxiv.org/pdf/1705.08790.pdf
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
            input_flat = input.view(-1)
            target_flat = target.view(-1)
            if ignore is not None:
                mask = target_flat != ignore
                input_flat = input_flat[mask]
                target_flat = target_flat[mask]
            loss += lovasz_hinge_flat(input_flat, target_flat)
        return loss / logits.size(0)
    else:
        loss = lovasz_hinge_flat(logits.view(-1), labels.view(-1))
        return loss


class LovaszHingeLoss(nn.Module):
    def __init__(self, per_image=True, ignore=None):
        super().__init__()
        self.per_image = per_image
        self.ignore = ignore

    def forward(self, input, target):
        """
        Args:
            input: Logits. Shape (B, 1, H, W) or list of (B, 1, H, W) for deep supervision.
            target: Binary mask. Shape (B, H, W) or (B, 1, H, W).
        """
        # Handle Deep Supervision (list of inputs)
        if isinstance(input, (list, tuple)):
            total_loss = 0
            for x in input:
                total_loss += self._compute_loss(x, target)
            return total_loss / len(input)
        else:
            return self._compute_loss(input, target)

    def _compute_loss(self, input, target):
        # Squeeze channel dim if present in input: (B, 1, H, W) -> (B, H, W)
        if input.dim() == 4 and input.size(1) == 1:
            input = input.squeeze(1)

        # Squeeze channel dim if present in target: (B, 1, H, W) -> (B, H, W)
        if target.dim() == 4 and target.size(1) == 1:
            target = target.squeeze(1)

        return lovasz_hinge(input, target, per_image=self.per_image, ignore=self.ignore)


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # Apply sigmoid to logits to get probabilities
        probs = torch.sigmoid(logits)

        # Flatten
        probs = probs.view(-1)
        targets = targets.view(-1)

        intersection = (probs * targets).sum()
        union = probs.sum() + targets.sum()

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, input, target):
        """
        Args:
            input: Logits. Shape (B, 1, H, W) or list of (B, 1, H, W).
            target: Binary mask. Shape (B, H, W) or (B, 1, H, W).
        """
        # Handle Deep Supervision
        if isinstance(input, (list, tuple)):
            total_loss = 0
            for x in input:
                total_loss += self._compute_loss(x, target)
            return total_loss / len(input)
        else:
            return self._compute_loss(input, target)

    def _compute_loss(self, input, target):
        # Ensure target has same shape as input for BCE
        if target.shape != input.shape:
            # If target is (B, H, W) and input is (B, 1, H, W), unsqueeze target
            if target.dim() == 3 and input.dim() == 4:
                target = target.unsqueeze(1)

        bce_loss = self.bce(input, target)
        dice_loss = self.dice(input, target)

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss
