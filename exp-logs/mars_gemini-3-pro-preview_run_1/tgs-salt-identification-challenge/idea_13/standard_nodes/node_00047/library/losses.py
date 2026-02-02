import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class BCEDiceLoss(nn.Module):
    """
    Combination of Binary Cross Entropy and Sample-Wise Dice Loss.
    Used for the initial training phase to establish robust feature extraction.

    The Dice component is calculated per-image (sample-wise) and averaged,
    which correctly handles empty masks by producing a high score (low loss)
    when both prediction and target are empty.
    """

    def __init__(self, bce_weight=1.0, dice_weight=1.0, smooth=1.0):
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Args:
            logits: (N, 1, H, W) raw model outputs (before sigmoid)
            targets: (N, 1, H, W) binary ground truth masks (0.0 or 1.0)
        """
        # 1. Binary Cross Entropy (Pixel-wise)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets)

        # 2. Sample-Wise Dice Loss
        probs = torch.sigmoid(logits)

        # Flatten spatial dimensions: (N, 1, H, W) -> (N, H*W)
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)

        intersection = (probs_flat * targets_flat).sum(dim=1)
        union = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)

        # Calculate Dice score per image
        # If both intersection and union are 0 (empty pred and empty target), score is 1.0 due to smooth
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Dice Loss is 1 - Dice Score
        dice_loss = 1.0 - dice_score.mean()

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class LovaszHingeLoss(nn.Module):
    """
    Lovasz-Hinge Loss for binary segmentation.
    Directly optimizes the Jaccard index (IoU) using the Lovasz extension.
    Used for the fine-tuning phase to maximize the competition metric.

    Includes an optional weighted BCE term for stability.
    """

    def __init__(self, bce_weight=1.0):
        super(LovaszHingeLoss, self).__init__()
        self.bce_weight = bce_weight

    def forward(self, logits, targets):
        """
        Args:
            logits: (N, 1, H, W) raw model outputs (before sigmoid)
            targets: (N, 1, H, W) binary ground truth masks (0.0 or 1.0)
        """
        # Lovasz Hinge Loss (calculated per image)
        lovasz = lovasz_hinge(logits, targets, per_image=True)

        # Optional BCE term
        if self.bce_weight > 0:
            bce = F.binary_cross_entropy_with_logits(logits, targets)
            return lovasz + self.bce_weight * bce

        return lovasz


# -----------------------------------------------------------------------------
# Lovasz-Softmax / Hinge Helper Functions
# Adapted from: https://github.com/bermanmaxim/LovaszSoftmax
# -----------------------------------------------------------------------------


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Jaccard loss w.r.t the sorted error
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


def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each pixel (or [N, P] if per_image)
      labels: [P] Tensor, binary ground truth labels (0 or 1) (or [N, P])
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        loss = 0
        for logit, label in zip(logits, labels):
            logit_flat = logit.view(-1)
            label_flat = label.view(-1)
            if ignore is not None:
                valid = label_flat != ignore
                logit_flat = logit_flat[valid]
                label_flat = label_flat[valid]
            loss += lovasz_hinge_flat(logit_flat, label_flat)
        return loss / logits.size(0)
    else:
        loss = lovasz_hinge_flat(logits.view(-1), labels.view(-1))
        return loss


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each pixel
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
