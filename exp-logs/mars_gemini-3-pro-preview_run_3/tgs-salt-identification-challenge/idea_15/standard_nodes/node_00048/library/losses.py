import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class BCEDiceLoss(nn.Module):
    """
    Combination of Binary Cross Entropy and Soft Dice Loss.
    Used for Phase 1: Structural Warm-up.
    """

    def __init__(self, bce_weight=1.0, dice_weight=1.0, smooth=1.0):
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, 1, H, W) or (B, H, W) raw model output (before sigmoid).
            targets: (B, 1, H, W) or (B, H, W) ground truth masks [0, 1].
        """
        # Ensure targets match logits shape
        if logits.shape != targets.shape:
            targets = targets.view_as(logits)

        # 1. BCE Loss
        bce_loss = self.bce(logits, targets)

        # 2. Dice Loss
        logits_sigmoid = torch.sigmoid(logits)

        # Flatten for Dice calculation
        batch_size = logits.size(0)
        probs_flat = logits_sigmoid.view(batch_size, -1)
        targets_flat = targets.view(batch_size, -1)

        intersection = (probs_flat * targets_flat).sum(dim=1)
        union = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)

        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice_score.mean()

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Jaccard loss w.r.t the errors.
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.cumsum(0)
    union = gts + (1 - gt_sorted).cumsum(0)
    jaccard = 1.0 - intersection / union

    # Handle potential division by zero or empty sets if necessary,
    # though usually covered by logic.

    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]

    return jaccard


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss on flattened tensors.

    Args:
        logits: [P] Float, logits of the prediction
        labels: [P] Float, binary labels (0 or 1)
    """
    # Treat labels as binary {0, 1}
    # If labels are float probabilities, we threshold them or treat them as hard labels for sorting
    # Standard Lovasz assumes binary ground truth.

    # Signs: -1 if label=0, +1 if label=1
    signs = 2.0 * labels - 1.0
    errors = 1.0 - logits * Variable(signs)

    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data

    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)

    loss = torch.dot(F.elu(errors_sorted) + 1, Variable(grad))
    return loss


class LovaszHingeLoss(nn.Module):
    """
    Lovasz-Hinge loss for optimizing the Jaccard index (IoU).
    Used for Phase 2: Metric Fine-tuning.
    Calculated per-image as per the strategy.
    """

    def __init__(self, per_image=True, ignore_index=None):
        super(LovaszHingeLoss, self).__init__()
        self.per_image = per_image
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, 1, H, W) or (B, H, W) raw model output.
            targets: (B, 1, H, W) or (B, H, W) ground truth masks.
        """
        if logits.shape != targets.shape:
            targets = targets.view_as(logits)

        if self.per_image:
            loss = 0
            batch_size = logits.size(0)
            for i in range(batch_size):
                # Flatten single image
                logit_flat = logits[i].view(-1)
                target_flat = targets[i].view(-1)

                # Filter ignore_index if specified
                if self.ignore_index is not None:
                    valid = target_flat != self.ignore_index
                    if valid.sum() == 0:
                        continue
                    logit_flat = logit_flat[valid]
                    target_flat = target_flat[valid]

                loss += lovasz_hinge_flat(logit_flat, target_flat)

            return loss / batch_size
        else:
            # Flatten entire batch
            logits_flat = logits.view(-1)
            targets_flat = targets.view(-1)

            if self.ignore_index is not None:
                valid = targets_flat != self.ignore_index
                logits_flat = logits_flat[valid]
                targets_flat = targets_flat[valid]

            return lovasz_hinge_flat(logits_flat, targets_flat)
