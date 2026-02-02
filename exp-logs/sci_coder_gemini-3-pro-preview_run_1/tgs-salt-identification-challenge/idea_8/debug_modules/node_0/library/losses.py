import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# -------------------------------------------------------------------------
# Lovasz-Softmax / Hinge Helper Functions
# -------------------------------------------------------------------------


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Jaccard loss w.r.t the sorted error.
    See Berman et al. CVPR 2018.
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
    Binary Lovasz hinge loss on flattened inputs.
    Args:
        logits: [P] Logits
        labels: [P] Binary labels (0 or 1)
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


# -------------------------------------------------------------------------
# Core Loss Modules
# -------------------------------------------------------------------------


class SampleWiseDiceLoss(nn.Module):
    """
    Calculates Dice Loss per image (sample-wise) and averages the result.
    This ensures that empty masks are handled correctly compared to batch-wise aggregation.
    Formula: 1 - (2 * |A n B| + smooth) / (|A| + |B| + smooth)
    """

    def __init__(self, smooth=1.0):
        super(SampleWiseDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Args:
            logits: (N, C, H, W) or (N, H, W) raw scores (no sigmoid applied).
            targets: (N, C, H, W) or (N, H, W) binary ground truth (0 or 1).
        """
        # Apply sigmoid to convert logits to probabilities
        probs = torch.sigmoid(logits)

        # Flatten spatial dimensions: (N, -1)
        # We preserve the batch dimension (N) to calculate per-sample Dice
        if probs.dim() > 2:
            probs = probs.view(probs.size(0), -1)

        if targets.dim() > 2:
            targets = targets.view(targets.size(0), -1)

        # Calculate intersection and sums per sample
        intersection = (probs * targets).sum(dim=1)
        cardinality = probs.sum(dim=1) + targets.sum(dim=1)

        # Calculate Dice score per sample
        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

        # Loss is 1 - mean dice score across the batch
        return 1.0 - dice_score.mean()


class LovaszHingeLoss(nn.Module):
    """
    Lovasz-Hinge Loss for binary segmentation.
    Optimizes the Jaccard index directly using the Lovasz extension.
    """

    def __init__(self):
        super(LovaszHingeLoss, self).__init__()

    def forward(self, logits, targets):
        """
        Args:
            logits: (N, C, H, W) or (N, H, W) raw scores.
            targets: (N, C, H, W) or (N, H, W) binary ground truth.
        """
        # Squeeze channel dim if present (N, 1, H, W) -> (N, H, W)
        if logits.dim() == 4 and logits.size(1) == 1:
            logits = logits.squeeze(1)
        if targets.dim() == 4 and targets.size(1) == 1:
            targets = targets.squeeze(1)

        # Flatten all dimensions to apply Lovasz over the entire batch
        # This optimizes the global IoU of the batch
        logits_flat = logits.view(-1)
        targets_flat = targets.view(-1)

        loss = lovasz_hinge_flat(logits_flat, targets_flat)
        return loss


# -------------------------------------------------------------------------
# Composite Loss Modules (Curriculum Learning)
# -------------------------------------------------------------------------


class Phase1Loss(nn.Module):
    """
    Composite loss for Phase 1 (Cycles 1 & 2).
    Combines Binary Cross Entropy (BCE) and Sample-Wise Dice Loss.
    Goal: Establish robust feature extraction and stable convergence.
    """

    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1.0):
        super(Phase1Loss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = SampleWiseDiceLoss(smooth=smooth)

    def forward(self, logits, targets):
        bce = self.bce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        return self.bce_weight * bce + self.dice_weight * dice


class Phase2Loss(nn.Module):
    """
    Composite loss for Phase 2 (Cycle 3).
    Combines Binary Cross Entropy (BCE) and Lovasz-Hinge Loss.
    Goal: Direct optimization of the Jaccard index (IoU) for fine-tuning.
    """

    def __init__(self, bce_weight=0.5, lovasz_weight=0.5):
        super(Phase2Loss, self).__init__()
        self.bce_weight = bce_weight
        self.lovasz_weight = lovasz_weight
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.lovasz_loss = LovaszHingeLoss()

    def forward(self, logits, targets):
        bce = self.bce_loss(logits, targets)
        lovasz = self.lovasz_loss(logits, targets)
        return self.bce_weight * bce + self.lovasz_weight * lovasz
