import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config

# -------------------------------------------------------------------------
# Lovasz-Hinge Loss Implementation
# References: https://github.com/bermanmaxim/LovaszSoftmax
# -------------------------------------------------------------------------


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Jaccard loss with respect to the sorted errors.
    See Berman et al. CVPR 2018.
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union

    # To avoid nan/inf if union is 0 (both prediction and target are empty)
    # However, in the hinge loss formulation, this is handled by the valid range of errors.
    # We pad the beginning to compute differences.
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss for a single flattened image.

    Args:
        logits: [P] Float, logits at each pixel.
        labels: [P] Float or Int, binary ground truth masks (0 or 1).

    Returns:
        Scalar loss.
    """
    if len(labels) == 0:
        # Should typically not happen with fixed image sizes
        return logits.sum() * 0.0

    signs = 2.0 * labels.float() - 1.0
    errors = 1.0 - logits * signs
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]

    grad = lovasz_grad(gt_sorted)

    # ELU + 1 is the hinge used in the paper
    loss = torch.dot(F.elu(errors_sorted) + 1, grad)
    return loss


class LovaszHingeLoss(nn.Module):
    """
    Lovasz-Hinge loss for binary segmentation.
    Optimizes the Jaccard index directly.
    """

    def __init__(self, per_image=True):
        super(LovaszHingeLoss, self).__init__()
        self.per_image = per_image

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, 1, H, W) or (B, H, W)
            targets: (B, 1, H, W) or (B, H, W)
        """
        # Squeeze channel dim if present
        if logits.dim() > 3:
            logits = logits.squeeze(1)
        if targets.dim() > 3:
            targets = targets.squeeze(1)

        # Ensure targets are float for consistency, though lovasz helper handles it
        targets = targets.float()

        if self.per_image:
            loss = 0
            batch_size = logits.size(0)
            for i in range(batch_size):
                # Flatten spatial dimensions
                l_flat = logits[i].view(-1)
                t_flat = targets[i].view(-1)
                loss += lovasz_hinge_flat(l_flat, t_flat)
            return loss / batch_size
        else:
            l_flat = logits.view(-1)
            t_flat = targets.view(-1)
            return lovasz_hinge_flat(l_flat, t_flat)


# -------------------------------------------------------------------------
# Sample-Wise Dice Loss Implementation
# -------------------------------------------------------------------------


class SampleWiseDiceLoss(nn.Module):
    """
    Dice Loss calculated per sample and averaged.
    Formula: 1 - (2 * I + smooth) / (U + smooth)
    """

    def __init__(self, smooth=1e-5):
        super(SampleWiseDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, 1, H, W) or (B, H, W) - Raw logits (no sigmoid applied yet)
            targets: (B, 1, H, W) or (B, H, W) - Binary masks
        """
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits)

        # Ensure correct shapes (B, H, W)
        if probs.dim() > 3:
            probs = probs.squeeze(1)
        if targets.dim() > 3:
            targets = targets.squeeze(1)

        batch_size = probs.size(0)

        # Flatten spatial dims: (B, H*W)
        probs_flat = probs.view(batch_size, -1)
        targets_flat = targets.view(batch_size, -1).float()

        intersection = (probs_flat * targets_flat).sum(dim=1)
        union = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        loss = 1.0 - dice

        return loss.mean()


# -------------------------------------------------------------------------
# Curriculum Loss Implementation
# -------------------------------------------------------------------------


class CurriculumLoss(nn.Module):
    """
    Curriculum Learning Loss Wrapper.

    Strategy:
    - Phase 1 (Epoch < CYCLE_1_END_EPOCH): BCE + Sample-Wise Dice
      Focus: Establishing structural convergence.

    - Phase 2 (Epoch >= CYCLE_1_END_EPOCH): BCE + Lovasz-Hinge
      Focus: Optimizing the Jaccard metric directly.
    """

    def __init__(self):
        super(CurriculumLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = SampleWiseDiceLoss()
        self.lovasz = LovaszHingeLoss(per_image=True)
        self.switch_epoch = Config.CYCLE_1_END_EPOCH

    def forward(self, logits, targets, epoch):
        """
        Args:
            logits: Model output logits.
            targets: Ground truth masks.
            epoch: Current training epoch (0-indexed).
        """
        # Calculate BCE (Common to both phases)
        # BCEWithLogitsLoss handles sigmoid internally
        bce_loss = self.bce(logits, targets.float())

        if epoch < self.switch_epoch:
            # Phase 1: BCE + Dice
            dice_loss = self.dice(logits, targets)
            return bce_loss + dice_loss
        else:
            # Phase 2: BCE + Lovasz
            lovasz_loss = self.lovasz(logits, targets)
            return bce_loss + lovasz_loss
