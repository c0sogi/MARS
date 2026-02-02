import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class DiceLoss(nn.Module):
    """
    Dice Coefficient Loss for binary segmentation.
    Formula: 1 - (2 * |A intersect B| + smooth) / (|A| + |B| + smooth)
    """

    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # Apply sigmoid to convert logits to probabilities
        probs = torch.sigmoid(logits)

        # Flatten predictions and targets
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()

        dice_score = (2.0 * intersection + self.smooth) / (
            probs_flat.sum() + targets_flat.sum() + self.smooth
        )

        return 1.0 - dice_score


class BCEDiceLoss(nn.Module):
    """
    Composite loss function combining Binary Cross Entropy (BCE) and Dice Loss.
    Used for the warm-up phase of training to ensure stable convergence.
    """

    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1.0):
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(smooth=smooth)

    def forward(self, logits, targets):
        # BCEWithLogitsLoss handles the sigmoid internally for numerical stability
        bce = self.bce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)

        return (self.bce_weight * bce) + (self.dice_weight * dice)


# -----------------------------------------------------------------------------
# Lovasz-Softmax / Lovasz-Hinge Implementation
# Reference: https://github.com/bermanmaxim/LovaszSoftmax
# -----------------------------------------------------------------------------


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Jaccard loss w.r.t the sorted errors.
    See Alg. 1 in https://arxiv.org/abs/1705.08790
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
    Binary Lovasz hinge loss for flattened inputs.

    Args:
        logits: [P] Variable, logits at each pixel (between -\infty and +\infty)
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


class LovaszHingeLoss(nn.Module):
    """
    Lovasz-Hinge loss for binary segmentation.
    Optimizes the Jaccard index (IoU) directly using the Lovasz extension.

    This loss is typically used for fine-tuning after the model has converged
    using BCE/Dice loss.
    """

    def __init__(self, per_image=True):
        """
        Args:
            per_image (bool): If True, computes the loss for each image independently
                              and averages the results (recommended for small batch sizes).
                              If False, computes the loss over the flattened batch.
        """
        super(LovaszHingeLoss, self).__init__()
        self.per_image = per_image

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): (B, 1, H, W) Raw model outputs (before sigmoid).
            targets (torch.Tensor): (B, 1, H, W) Binary ground truth masks (0 or 1).
        """
        if self.per_image:
            batch_size = logits.size(0)
            losses = []
            for i in range(batch_size):
                # Flatten each image in the batch
                logit_flat = logits[i].view(-1)
                target_flat = targets[i].view(-1)

                # Calculate Lovasz hinge loss for this image
                loss = lovasz_hinge_flat(logit_flat, target_flat)
                losses.append(loss)

            return sum(losses) / batch_size
        else:
            # Flatten the entire batch
            return lovasz_hinge_flat(logits.view(-1), targets.view(-1))
