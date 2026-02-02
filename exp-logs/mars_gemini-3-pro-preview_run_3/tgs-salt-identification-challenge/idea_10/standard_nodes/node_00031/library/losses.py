import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DiceLoss(nn.Module):
    """
    Dice coefficient loss for binary segmentation.
    Computes 1 - Dice Score.
    Expects logits as input.
    """

    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # Apply sigmoid to convert logits to probabilities
        probs = torch.sigmoid(logits)

        # Flatten label and prediction tensors
        probs = probs.view(-1)
        targets = targets.view(-1)

        intersection = (probs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (
            probs.sum() + targets.sum() + self.smooth
        )

        return 1.0 - dice


class BCEDiceLoss(nn.Module):
    """
    Combination of Binary Cross Entropy and Dice Loss.
    Used for the warm-up phase of training.
    """

    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1.0):
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(smooth=smooth)

    def forward(self, logits, targets):
        bce = self.bce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        return self.bce_weight * bce + self.dice_weight * dice


# -------------------------------------------------------------------------
# Lovasz-Hinge Loss Implementation
# Adapted from: https://github.com/bermanmaxim/LovaszSoftmax
# -------------------------------------------------------------------------


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
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


def lovasz_hinge(logits, labels, per_image=True):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth labels (0 or 1)
      per_image: compute the loss per image instead of per batch
    """
    if per_image:
        loss = 0
        batch_size = logits.size(0)
        for i in range(batch_size):
            # Flatten per image
            l = logits[i].view(-1)
            t = labels[i].view(-1)
            loss += lovasz_hinge_flat(l, t)
        return loss / batch_size
    else:
        return lovasz_hinge_flat(logits.view(-1), labels.view(-1))


class LovaszHingeLoss(nn.Module):
    """
    Lovasz-Hinge loss for optimizing Jaccard index directly.
    Used for the fine-tuning phase.
    """

    def __init__(self, per_image=True):
        super(LovaszHingeLoss, self).__init__()
        self.per_image = per_image

    def forward(self, logits, targets):
        """
        Args:
            logits: [B, 1, H, W] or [B, H, W]
            targets: [B, 1, H, W] or [B, H, W]
        """
        # Squeeze channel dimension if present (B, 1, H, W) -> (B, H, W)
        if logits.dim() > 3:
            logits = logits.squeeze(1)
        if targets.dim() > 3:
            targets = targets.squeeze(1)

        return lovasz_hinge(logits, targets, per_image=self.per_image)


# -------------------------------------------------------------------------
# Deep Supervision Wrapper
# -------------------------------------------------------------------------


class DeepSupervisionLoss(nn.Module):
    """
    Wrapper to apply a base loss function to multiple outputs from the model.
    Used for U-Net++ which returns a list of tensors [output_L1, output_L2, ..., output_final].
    """

    def __init__(self, base_loss, weights=None):
        """
        Args:
            base_loss: The loss module to apply (e.g., BCEDiceLoss or LovaszHingeLoss).
            weights: List of weights for each output. If None, equal weights are assumed.
                     For U-Net++, standard weights might be [0.1, 0.1, 0.1, 1.0].
        """
        super(DeepSupervisionLoss, self).__init__()
        self.base_loss = base_loss
        self.weights = weights

    def forward(self, preds, targets):
        # If preds is not a list/tuple (i.e., deep supervision disabled or inference), just apply base loss
        if not isinstance(preds, (list, tuple)):
            return self.base_loss(preds, targets)

        loss = 0.0
        num_outputs = len(preds)

        # Initialize weights if not provided
        if self.weights is None:
            weights = [1.0] * num_outputs
        else:
            weights = self.weights
            if len(weights) != num_outputs:
                # Fallback if weights length mismatch
                weights = [1.0] * num_outputs

        # Normalize weights to sum to 1 (optional, but good for stability)
        # Here we just sum the weighted losses

        for i, pred in enumerate(preds):
            # Targets are assumed to be the same for all scales
            # U-Net++ outputs are usually upsampled to input size, so no resizing of target needed
            loss += weights[i] * self.base_loss(pred, targets)

        return loss
