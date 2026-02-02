import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Jaccard loss w.r.t the sorted error
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
    Wrapper for Lovasz Hinge Loss.
    """

    def __init__(self):
        super(LovaszHingeLoss, self).__init__()

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C, H, W) or (B, H, W) - Raw scores (before sigmoid)
            targets: (B, C, H, W) or (B, H, W) - Binary masks (0 or 1)
        """
        # Flatten the tensors
        logits_flat = logits.reshape(-1)
        targets_flat = targets.reshape(-1)

        loss = lovasz_hinge_flat(logits_flat, targets_flat)
        return loss


class SampleWiseDiceLoss(nn.Module):
    """
    Calculates Dice Loss for each sample in the batch and averages them.
    This is preferred over batch-wise Dice for this task to correctly handle
    empty masks (images with no salt).
    """

    def __init__(self, smooth=1.0):
        super(SampleWiseDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, 1, H, W) - Raw scores
            targets: (B, 1, H, W) - Binary masks
        """
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits)

        # Flatten spatial dimensions: (B, H*W)
        batch_size = probs.shape[0]
        probs_flat = probs.view(batch_size, -1)
        targets_flat = targets.view(batch_size, -1)

        # Calculate intersection and sums per sample
        intersection = (probs_flat * targets_flat).sum(dim=1)
        sum_p = probs_flat.sum(dim=1)
        sum_t = targets_flat.sum(dim=1)

        # Calculate Dice coefficient per sample
        dice_score = (2.0 * intersection + self.smooth) / (sum_p + sum_t + self.smooth)

        # Loss is 1 - Mean Dice
        return 1.0 - dice_score.mean()


class CompoundLoss(nn.Module):
    """
    Consistent Compound Loss combining:
    1. Binary Cross Entropy (BCE)
    2. Sample-Wise Dice Loss
    3. Lovasz-Hinge Loss

    Weights are defined in Config.
    """

    def __init__(self):
        super(CompoundLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = SampleWiseDiceLoss()
        self.lovasz = LovaszHingeLoss()

        self.w_bce = Config.WEIGHT_BCE
        self.w_dice = Config.WEIGHT_DICE
        self.w_lovasz = Config.WEIGHT_LOVASZ

    def _compute_single_loss(self, logits, targets):
        loss_bce = self.bce(logits, targets)
        loss_dice = self.dice(logits, targets)
        loss_lovasz = self.lovasz(logits, targets)

        total_loss = (
            self.w_bce * loss_bce
            + self.w_dice * loss_dice
            + self.w_lovasz * loss_lovasz
        )

        metrics = {
            "bce": loss_bce.detach(),
            "dice": loss_dice.detach(),
            "lovasz": loss_lovasz.detach(),
            "total": total_loss.detach(),
        }
        return total_loss, metrics

    def forward(self, preds, targets):
        """
        Args:
            preds: (B, 1, H, W) or tuple of tensors for Deep Supervision
            targets: (B, 1, H, W)
        Returns:
            total_loss: scalar tensor
            metrics: dict containing individual loss components for logging
        """
        # Ensure targets are float for BCE
        targets = targets.float()

        if isinstance(preds, tuple):
            logits_final, aux_64, aux_32 = preds

            # Main Loss
            loss_final, metrics = self._compute_single_loss(logits_final, targets)

            # Aux Losses (Upsample aux outputs to target size)
            aux_64_up = F.interpolate(
                aux_64, size=targets.shape[2:], mode="bilinear", align_corners=True
            )
            aux_32_up = F.interpolate(
                aux_32, size=targets.shape[2:], mode="bilinear", align_corners=True
            )

            loss_aux_64, _ = self._compute_single_loss(aux_64_up, targets)
            loss_aux_32, _ = self._compute_single_loss(aux_32_up, targets)

            # Weighted Sum: 1.0, 0.5, 0.25 (Decaying weights for lower resolutions)
            total_loss = loss_final + 0.5 * loss_aux_64 + 0.25 * loss_aux_32

            metrics["total"] = total_loss.detach()
            return total_loss, metrics

        else:
            return self._compute_single_loss(preds, targets)
