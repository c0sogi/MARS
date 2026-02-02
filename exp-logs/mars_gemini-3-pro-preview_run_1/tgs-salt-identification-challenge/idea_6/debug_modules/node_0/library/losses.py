import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------------------------------------------------
# Lovasz-Hinge Loss Implementation
# References: https://github.com/bermanmaxim/LovaszSoftmax
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


class LovaszHingeLoss(nn.Module):
    """
    Lovasz-Hinge loss for binary segmentation.
    Optimizes the Jaccard index (IoU) directly.
    """

    def __init__(self, per_image=True):
        super(LovaszHingeLoss, self).__init__()
        self.per_image = per_image

    def forward(self, logits, targets):
        """
        Args:
            logits: (N, 1, H, W) or (N, H, W)
            targets: (N, 1, H, W) or (N, H, W)
        """
        # Unify shapes
        if logits.dim() == 4:
            logits = logits.squeeze(1)
        if targets.dim() == 4:
            targets = targets.squeeze(1)

        if self.per_image:
            loss = 0
            batch_size = logits.size(0)
            for i in range(batch_size):
                # Flatten per image
                l = logits[i].view(-1)
                t = targets[i].view(-1)
                loss += lovasz_hinge_flat(l, t)
            return loss / batch_size
        else:
            # Flatten batch
            return lovasz_hinge_flat(logits.view(-1), targets.view(-1))


# -------------------------------------------------------------------------
# Sample-Wise Dice Loss
# -------------------------------------------------------------------------


class SampleWiseDiceLoss(nn.Module):
    """
    Calculates Dice Loss per image and averages it.
    This ensures empty masks (common in salt data) are handled correctly.
    """

    def __init__(self, smooth=1.0):
        super(SampleWiseDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Args:
            logits: (N, 1, H, W) or (N, H, W) - Raw logits (no sigmoid applied yet)
            targets: (N, 1, H, W) or (N, H, W) - Binary targets (0 or 1)
        """
        # Unify shapes
        if logits.dim() == 4:
            logits = logits.squeeze(1)
        if targets.dim() == 4:
            targets = targets.squeeze(1)

        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits)

        # Flatten per sample: (N, H*W)
        batch_size = probs.size(0)
        probs_flat = probs.view(batch_size, -1)
        targets_flat = targets.view(batch_size, -1)

        # Calculate intersection and union per sample
        intersection = (probs_flat * targets_flat).sum(dim=1)
        # Cardinality (Sum of elements)
        cardinality = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)

        # Calculate Dice score per sample
        # Add smooth to numerator and denominator to handle 0/0 case (empty mask + empty pred = 1)
        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

        # Loss is 1 - Mean Dice
        return 1.0 - dice_score.mean()


# -------------------------------------------------------------------------
# Composite Losses
# -------------------------------------------------------------------------


class BCEDiceLoss(nn.Module):
    """
    Combination of BCEWithLogitsLoss and SampleWiseDiceLoss.
    Used for the initial training phase to establish convergence.
    """

    def __init__(self, bce_weight=1.0, dice_weight=1.0, smooth=1.0):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = SampleWiseDiceLoss(smooth=smooth)
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        # BCE expects float targets
        bce_loss = self.bce(logits, targets.float())
        dice_loss = self.dice(logits, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class LovaszBCELoss(nn.Module):
    """
    Combination of LovaszHingeLoss and BCEWithLogitsLoss.
    Used for the fine-tuning phase to optimize IoU directly while maintaining stability.
    """

    def __init__(self, lovasz_weight=1.0, bce_weight=0.1, per_image=True):
        super(LovaszBCELoss, self).__init__()
        self.lovasz = LovaszHingeLoss(per_image=per_image)
        self.bce = nn.BCEWithLogitsLoss()
        self.lovasz_weight = lovasz_weight
        self.bce_weight = bce_weight

    def forward(self, logits, targets):
        lovasz_loss = self.lovasz(logits, targets)
        bce_loss = self.bce(logits, targets.float())
        return self.lovasz_weight * lovasz_loss + self.bce_weight * bce_loss


# -------------------------------------------------------------------------
# Factory Function
# -------------------------------------------------------------------------


def get_loss_for_phase(current_epoch, switch_epoch=100):
    """
    Returns the appropriate loss function based on the training phase.

    Args:
        current_epoch (int): The current training epoch (0-indexed).
        switch_epoch (int): The epoch at which to switch from BCE+Dice to Lovasz+BCE.
                            Defaults to 100 (start of Cycle 3 in a 150 epoch schedule).

    Returns:
        nn.Module: The loss function module.
    """
    if current_epoch < switch_epoch:
        # Phase 1: Robust convergence with BCE + Dice
        return BCEDiceLoss(bce_weight=1.0, dice_weight=1.0)
    else:
        # Phase 2: Fine-tuning with Lovasz-Hinge + small BCE regularization
        return LovaszBCELoss(lovasz_weight=1.0, bce_weight=0.1, per_image=True)
