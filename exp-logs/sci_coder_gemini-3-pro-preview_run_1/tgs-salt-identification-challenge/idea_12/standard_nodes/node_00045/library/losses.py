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
    Computes gradient of the Jaccard loss w.r.t the sorted ground truth.
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union

    # To avoid 0/0 division if union is 0 (i.e. empty mask and all predictions correct)
    jaccard[union == 0] = 0.0

    gradients = jaccard[1:] - jaccard[:-1]
    return gradients


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss for a flat vector.

    Args:
        logits: (N,) logits
        labels: (N,) binary labels (0 or 1)
    """
    if len(labels) == 0:
        # valid for empty tensors
        return logits.sum() * 0.0

    signs = 2.0 * labels.float() - 1.0
    errors = 1.0 - logits * signs
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(
        F.relu(errors_sorted),
        torch.cat((grad, torch.tensor([0.0], device=grad.device))),
    )
    return loss


def lovasz_hinge(logits, labels, per_image=True):
    """
    Binary Lovasz hinge loss.

    Args:
        logits: [B, 1, H, W] or [B, H, W] logits
        labels: [B, H, W] or [B, 1, H, W] binary labels
        per_image: compute the loss per image instead of per batch
    """
    # Squeeze channel dim if present
    if logits.dim() > 3:
        logits = logits.squeeze(1)
    if labels.dim() > 3:
        labels = labels.squeeze(1)

    if per_image:
        loss = 0
        for input_i, target_i in zip(logits, labels):
            loss += lovasz_hinge_flat(input_i.flatten(), target_i.flatten())
        return loss / logits.size(0)
    else:
        return lovasz_hinge_flat(logits.flatten(), labels.flatten())


class LovaszHingeLoss(nn.Module):
    def __init__(self, per_image=True):
        super().__init__()
        self.per_image = per_image

    def forward(self, logits, targets):
        return lovasz_hinge(logits, targets, per_image=self.per_image)


# -------------------------------------------------------------------------
# Sample-Wise Dice Loss Implementation
# -------------------------------------------------------------------------


class SampleWiseDiceLoss(nn.Module):
    """
    Computes the Dice Loss for each image in the batch independently and averages the results.
    This handles empty masks correctly (Dice=1 if both pred and target are empty),
    unlike batch-wise Dice which aggregates pixels first.
    """

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, 1, H, W) raw logits from the model
            targets: (B, 1, H, W) or (B, H, W) binary ground truth
        """
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits)

        # Ensure targets match shape
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)

        # Flatten spatial dimensions: (B, 1, H, W) -> (B, H*W)
        batch_size = probs.size(0)
        probs_flat = probs.view(batch_size, -1)
        targets_flat = targets.view(batch_size, -1)

        # Calculate intersection and sums per sample
        intersection = (probs_flat * targets_flat).sum(dim=1)
        union = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)

        # Dice coefficient per sample
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Return 1 - mean(dice)
        return 1.0 - dice_score.mean()


# -------------------------------------------------------------------------
# Consistent Compound Loss Implementation
# -------------------------------------------------------------------------


class ConsistentCompoundLoss(nn.Module):
    """
    Aggregates BCE, Sample-Wise Dice, and Lovasz-Hinge loss.
    L = w_bce * BCE + w_dice * Dice + w_lovasz * Lovasz
    """

    def __init__(self):
        super().__init__()
        self.weight_bce = Config.WEIGHT_BCE
        self.weight_dice = Config.WEIGHT_DICE
        self.weight_lovasz = Config.WEIGHT_LOVASZ

        self.bce = nn.BCEWithLogitsLoss()
        self.dice = SampleWiseDiceLoss(smooth=1.0)
        self.lovasz = LovaszHingeLoss(per_image=True)

    def forward(self, logits, targets):
        # Ensure targets are float for BCE and Dice
        targets = targets.float()

        # BCEWithLogits expects (B, C, H, W) or broadcastable.
        # If targets is (B, H, W), unsqueeze it.
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)

        loss_bce = self.bce(logits, targets)
        loss_dice = self.dice(logits, targets)

        # Lovasz expects squeezed inputs typically, but our wrapper handles it.
        # We pass the same tensors.
        loss_lovasz = self.lovasz(logits, targets)

        total_loss = (
            self.weight_bce * loss_bce
            + self.weight_dice * loss_dice
            + self.weight_lovasz * loss_lovasz
        )

        return total_loss
