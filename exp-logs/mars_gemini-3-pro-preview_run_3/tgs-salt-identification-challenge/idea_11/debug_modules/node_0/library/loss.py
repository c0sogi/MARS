import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BCEDiceLoss(nn.Module):
    """
    Combination of Binary Cross Entropy Loss and Dice Loss.
    Used for the warm-up phase of training to establish convergence.
    """

    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1.0):
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Model logits of shape (B, C, H, W) or (B, H, W).
            targets (torch.Tensor): Ground truth masks of shape (B, C, H, W) or (B, H, W).
                                    Values should be 0 or 1.
        """
        # Ensure inputs and targets have the same shape
        if inputs.shape != targets.shape:
            targets = targets.view_as(inputs)

        # Flatten for BCE and Dice calculation
        inputs_flat = inputs.contiguous().view(-1)
        targets_flat = targets.contiguous().view(-1)

        # Binary Cross Entropy
        bce_loss = F.binary_cross_entropy_with_logits(inputs_flat, targets_flat.float())

        # Dice Loss
        probs = torch.sigmoid(inputs_flat)
        intersection = (probs * targets_flat).sum()
        dice_score = (2.0 * intersection + self.smooth) / (
            probs.sum() + targets_flat.sum() + self.smooth
        )
        dice_loss = 1.0 - dice_score

        # Weighted Combination
        total_loss = (self.bce_weight * bce_loss) + (self.dice_weight * dice_loss)

        return total_loss


class LovaszHingeLoss(nn.Module):
    """
    Lovasz-Hinge Loss for binary segmentation.
    Optimizes the Jaccard index (IoU) directly.
    Used for the fine-tuning phase.

    Reference: https://github.com/bermanmaxim/LovaszSoftmax
    """

    def __init__(self, per_image=True, ignore=None):
        super(LovaszHingeLoss, self).__init__()
        self.per_image = per_image
        self.ignore = ignore

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Model logits of shape (B, C, H, W) or (B, H, W).
            targets (torch.Tensor): Ground truth masks of shape (B, C, H, W) or (B, H, W).
        """
        if inputs.shape != targets.shape:
            targets = targets.view_as(inputs)

        return lovasz_hinge(
            inputs, targets, per_image=self.per_image, ignore=self.ignore
        )


# -----------------------------------------------------------------------------
# Lovasz-Softmax / Hinge Helper Functions
# -----------------------------------------------------------------------------


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted.float()).cumsum(0)
    jaccard = 1.0 - intersection / union

    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]

    return jaccard


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
        # Iterate over batch dimension
        batch_size = logits.size(0)
        for i in range(batch_size):
            # Flatten spatial dimensions for this image
            loss += lovasz_hinge_flat(
                *flatten_binary_scores(
                    logits[i].unsqueeze(0), labels[i].unsqueeze(0), ignore
                )
            )
        return loss / batch_size
    else:
        return lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss on flattened inputs
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

    # The loss is the dot product of the sorted errors (passed through ReLU) and the gradient
    loss = torch.dot(F.relu(errors_sorted), grad)
    return loss


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.contiguous().view(-1)
    labels = labels.contiguous().view(-1)
    if ignore is None:
        return scores, labels
    valid = labels != ignore
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels
