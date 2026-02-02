import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# ---------------------------
# Lovasz-Softmax Helpers
# ---------------------------


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Jaccard loss with respect to the errors
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
      logits: [P] Variable, logits of the prediction
      labels: [P] Tensor, binary ground truth labels (0 or 1)
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


# ---------------------------
# Loss Classes
# ---------------------------


class BCEDiceLoss(nn.Module):
    """
    Hybrid loss combining Binary Cross Entropy and Dice Loss.
    Used for the warm-up phase of training to establish convergence.
    """

    def __init__(self, smooth=1.0, bce_weight=0.5):
        super(BCEDiceLoss, self).__init__()
        self.smooth = smooth
        self.bce_weight = bce_weight

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C, H, W) or (B, H, W) Raw model outputs (before sigmoid)
            targets: (B, C, H, W) or (B, H, W) Binary ground truth masks
        """
        # Flatten inputs
        logits_flat = logits.view(-1)
        targets_flat = targets.view(-1)

        # BCE Loss (with logits for numerical stability)
        bce_loss = F.binary_cross_entropy_with_logits(logits_flat, targets_flat)

        # Dice Loss
        probs = torch.sigmoid(logits_flat)
        intersection = (probs * targets_flat).sum()
        dice_score = (2.0 * intersection + self.smooth) / (
            probs.sum() + targets_flat.sum() + self.smooth
        )
        dice_loss = 1.0 - dice_score

        return self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss


class LovaszHingeLoss(nn.Module):
    """
    Lovasz-Hinge loss for optimizing the Jaccard index.
    Crucial for the fine-tuning phase. Includes specific stability fixes for AMP.
    """

    def __init__(self, per_image=True):
        super(LovaszHingeLoss, self).__init__()
        self.per_image = per_image

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C, H, W) Raw model outputs
            targets: (B, C, H, W) Binary ground truth masks
        """
        # STABILITY FIX: Explicitly cast to float32.
        # The cumsum operation in lovasz_grad is unstable in FP16, leading to NaNs.
        logits = logits.float()
        targets = targets.float()

        if self.per_image:
            loss = 0.0
            batch_size = logits.size(0)
            for i in range(batch_size):
                # Flatten spatial dimensions: (C, H, W) -> (P)
                logit_flat = logits[i].view(-1)
                target_flat = targets[i].view(-1)
                loss += lovasz_hinge_flat(logit_flat, target_flat)
            return loss / batch_size
        else:
            logit_flat = logits.view(-1)
            target_flat = targets.view(-1)
            return lovasz_hinge_flat(logit_flat, target_flat)


class SoftBCELoss(nn.Module):
    """
    Binary Cross Entropy Loss that accepts soft targets (probabilities).
    Used for the semi-supervised self-training phase.
    """

    def __init__(self, reduction="mean"):
        super(SoftBCELoss, self).__init__()
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C, H, W) Raw model outputs
            targets: (B, C, H, W) Soft target probabilities [0, 1]
        """
        # BCEWithLogitsLoss natively supports soft targets in PyTorch
        return F.binary_cross_entropy_with_logits(
            logits, targets, reduction=self.reduction
        )
