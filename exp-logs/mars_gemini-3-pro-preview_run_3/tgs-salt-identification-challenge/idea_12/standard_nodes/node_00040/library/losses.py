import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice coefficient loss for binary segmentation.
    Computes 1 - Dice.
    """

    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits)

        # Flatten
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
    Supports Deep Supervision (list of inputs).
    """

    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1.0):
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth)

    def _compute_single_loss(self, logits, targets):
        # Ensure targets match logits shape (B, 1, H, W)
        if targets.dim() != logits.dim():
            targets = targets.unsqueeze(1)

        targets = targets.float()

        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss

    def forward(self, inputs, targets):
        # Handle Deep Supervision (list of outputs)
        if isinstance(inputs, (list, tuple)):
            loss = 0.0
            for item in inputs:
                loss += self._compute_single_loss(item, targets)
            return loss / len(inputs)
        else:
            return self._compute_single_loss(inputs, targets)


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
    Lovasz-Hinge loss for optimizing Jaccard index.
    Used for the fine-tuning phase of training.
    Supports Deep Supervision (list of inputs).
    """

    def __init__(self):
        super(LovaszHingeLoss, self).__init__()

    def _compute_single_loss(self, logits, targets):
        # Ensure targets match logits shape/type
        if targets.dim() != logits.dim():
            targets = targets.unsqueeze(1)

        targets = targets.float()

        # Calculate loss per sample to align with mean-based metrics (Cite 00023)
        loss = 0.0
        batch_size = logits.size(0)

        for i in range(batch_size):
            logits_flat = logits[i].view(-1)
            targets_flat = targets[i].view(-1)
            loss += lovasz_hinge_flat(logits_flat, targets_flat)

        return loss / batch_size

    def forward(self, inputs, targets):
        # Handle Deep Supervision (list of outputs)
        if isinstance(inputs, (list, tuple)):
            loss = 0.0
            for item in inputs:
                loss += self._compute_single_loss(item, targets)
            return loss / len(inputs)
        else:
            return self._compute_single_loss(inputs, targets)
