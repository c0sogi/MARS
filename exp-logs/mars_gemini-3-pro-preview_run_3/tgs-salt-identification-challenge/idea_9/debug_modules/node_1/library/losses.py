import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LovaszHingeLoss(nn.Module):
    """
    Lovasz-Hinge loss for binary segmentation.
    Optimizes the Jaccard index (IoU) directly using the Lovasz extension.
    """

    def __init__(self, per_image=True, ignore_index=None):
        super().__init__()
        self.per_image = per_image
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, 1, H, W) or (B, H, W) logits from the model.
            targets: (B, 1, H, W) or (B, H, W) binary ground truth masks.
        """
        # Squeeze channel dimension if present (B, 1, H, W) -> (B, H, W)
        if logits.dim() == 4 and logits.size(1) == 1:
            logits = logits.squeeze(1)
        if targets.dim() == 4 and targets.size(1) == 1:
            targets = targets.squeeze(1)

        return lovasz_hinge(
            logits, targets, per_image=self.per_image, ignore=self.ignore_index
        )


class BCEDiceLoss(nn.Module):
    """
    Combination of Binary Cross Entropy and Dice Loss.
    Used for the warm-up phase of training.
    """

    def __init__(self, bce_weight=0.5, smooth=1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        # BCE Loss
        bce_loss = self.bce(logits, targets)

        # Dice Loss
        # Apply sigmoid to convert logits to probabilities
        probs = torch.sigmoid(logits)

        # Flatten for Dice calculation
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            probs_flat.sum() + targets_flat.sum() + self.smooth
        )

        # Combined Loss
        # Dice loss is (1 - dice_coefficient)
        return self.bce_weight * bce_loss + (1 - self.bce_weight) * (1 - dice)


class DeepSupervisionLoss(nn.Module):
    """
    Wrapper to apply a specific loss function to multiple outputs
    from a Deep Supervision architecture (e.g., U-Net++).
    """

    def __init__(self, loss_fn, weights=None):
        super().__init__()
        self.loss_fn = loss_fn
        self.weights = weights

    def forward(self, outputs, targets):
        # If outputs is a list/tuple (Deep Supervision active)
        if isinstance(outputs, (list, tuple)):
            loss = 0
            num_outputs = len(outputs)

            # Use uniform weights if none provided
            if self.weights is None:
                ws = [1.0 / num_outputs] * num_outputs
            else:
                ws = self.weights
                assert (
                    len(ws) == num_outputs
                ), "Length of weights must match number of outputs"

            for output, w in zip(outputs, ws):
                loss += w * self.loss_fn(output, targets)
            return loss
        else:
            # Single output case
            return self.loss_fn(outputs, targets)


# --- Lovasz Hinge Helper Functions ---
# Adapted from: https://github.com/bermanmaxim/LovaszSoftmax


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
        for input, target in zip(logits, labels):
            loss = loss + lovasz_hinge_flat(
                *flatten_binary_scores(input.unsqueeze(0), target.unsqueeze(0), ignore)
            )
        return loss / logits.size(0)
    else:
        return lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss on flattened responses
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
