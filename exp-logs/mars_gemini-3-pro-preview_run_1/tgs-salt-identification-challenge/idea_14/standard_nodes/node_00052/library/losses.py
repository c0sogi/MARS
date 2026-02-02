import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# -------------------------------------------------------------------------
# Lovasz-Hinge Loss Implementation
# Adapted from: https://github.com/bermanmaxim/LovaszSoftmax
# -------------------------------------------------------------------------


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension of intersection-over-union metric
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.cumsum(0)
    union = gts + (1 - gt_sorted).cumsum(0)
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
      logits: [P] Variable, logits at each pixel (between -\infty and +\infty)
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
      labels: [B, H, W] Tensor, binary ground truth labels (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        loss = torch.mean(
            torch.stack(
                [
                    lovasz_hinge_flat(
                        *flatten_binary_scores(
                            log.unsqueeze(0), lab.unsqueeze(0), ignore
                        )
                    )
                    for log, lab in zip(logits, labels)
                ]
            )
        )
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss


# -------------------------------------------------------------------------
# Dice Loss Implementation
# -------------------------------------------------------------------------


class DiceLoss(nn.Module):
    """
    Sample-wise Soft Dice Loss.
    Calculates Dice score for each image in the batch and averages them.
    Expects logits as input.
    """

    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Args:
            logits: (N, 1, H, W) or (N, H, W)
            targets: (N, 1, H, W) or (N, H, W)
        """
        # Apply sigmoid to convert logits to probabilities
        probs = torch.sigmoid(logits)

        # Flatten per sample: (N, -1)
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (probs * targets).sum(1)
        union = probs.sum(1) + targets.sum(1)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Return 1 - Dice
        return 1 - dice.mean()


# -------------------------------------------------------------------------
# Curriculum Loss Implementation
# -------------------------------------------------------------------------


class CurriculumLoss(nn.Module):
    """
    Composite Loss with Intra-Cycle Curriculum.

    Structure:
    - Base: BCEWithLogitsLoss + DiceLoss
    - Curriculum: LovaszHingeLoss is added with dynamic weighting based on the epoch.

    Schedule:
    - Cycle 1 (Epochs 0-49): Lovasz Weight = 0
    - Cycle 2 (Epochs 50-99) & Cycle 3 (Epochs 100-149):
        - First 35 epochs of cycle: Lovasz Weight = 0
        - Last 15 epochs of cycle: Linear ramp from 0.0 to 0.5
    """

    def __init__(self):
        super(CurriculumLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

        # Hyperparameters for schedule
        self.cycle_length = 50
        self.ramp_start_epoch = 35  # Relative to cycle start
        self.max_lovasz_weight = 1.0
        self.ramp_duration = 15.0

    def get_lovasz_weight(self, epoch):
        # Determine which cycle we are in (0-indexed)
        cycle_idx = epoch // self.cycle_length

        # Determine epoch position within the current cycle (0-49)
        epoch_in_cycle = epoch % self.cycle_length

        # Cycle 1 (Index 0): Pure BCE+Dice
        if cycle_idx == 0:
            return 0.0

        # Cycle 2+ (Index > 0): Apply Curriculum
        # Phase A: First 35 epochs
        if epoch_in_cycle < self.ramp_start_epoch:
            return 0.0

        # Phase B: Last 15 epochs -> Ramp up
        # Calculate progress from 0.0 to 1.0 (approx)
        steps_past_start = epoch_in_cycle - self.ramp_start_epoch
        progress = steps_past_start / self.ramp_duration

        weight = self.max_lovasz_weight * progress

        # Clip to ensure we don't exceed max weight (though math holds)
        return min(weight, self.max_lovasz_weight)

    def forward(self, logits, targets, epoch):
        """
        Args:
            logits: (N, 1, H, W) Raw scores
            targets: (N, 1, H, W) Binary mask
            epoch: (int) Current training epoch (0-indexed)
        """
        # 1. Base Loss (BCE + Dice)
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        total_loss = bce_loss + dice_loss

        # 2. Curriculum Lovasz Loss
        w_lovasz = self.get_lovasz_weight(epoch)

        if w_lovasz > 0:
            # Lovasz Hinge expects logits.
            # It usually expects (N, H, W) or (N, C, H, W).
            # We squeeze the channel dim if it's 1 for consistency with lovasz signature
            if logits.dim() == 4 and logits.size(1) == 1:
                logits_sq = logits.squeeze(1)
                targets_sq = targets.squeeze(1)
            else:
                logits_sq = logits
                targets_sq = targets

            lov_loss = lovasz_hinge(logits_sq, targets_sq, per_image=True)
            total_loss += w_lovasz * lov_loss

        return total_loss
