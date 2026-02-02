import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# ---------------------------------------------------------------------------
# Lovasz-Hinge Loss Helpers
# Adapted from https://github.com/bermanmaxim/LovaszSoftmax
# ---------------------------------------------------------------------------


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
      ignore: void class id
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


def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth labels (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        loss = 0
        for input_tensor, target_tensor in zip(logits, labels):
            input_tensor = input_tensor.view(-1)
            target_tensor = target_tensor.view(-1)
            if ignore is not None:
                valid = target_tensor != ignore
                input_tensor = input_tensor[valid]
                target_tensor = target_tensor[valid]
            loss += lovasz_hinge_flat(input_tensor, target_tensor)
        return loss / logits.size(0)
    else:
        logits = logits.view(-1)
        labels = labels.view(-1)
        if ignore is not None:
            valid = labels != ignore
            logits = logits[valid]
            labels = labels[valid]
        return lovasz_hinge_flat(logits, labels)


# ---------------------------------------------------------------------------
# Loss Classes
# ---------------------------------------------------------------------------


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # Flatten inputs
        logits = logits.view(-1)
        targets = targets.view(-1)

        # Apply sigmoid to convert logits to probabilities
        probs = torch.sigmoid(logits)

        intersection = (probs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (
            probs.sum() + targets.sum() + self.smooth
        )

        return 1 - dice


class BCEDiceLoss(nn.Module):
    """
    Weighted combination of Binary Cross Entropy and Dice Loss.
    Used for the warm-up phase of training.
    """

    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits, targets):
        # Ensure targets are float for BCE
        targets = targets.float()

        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class LovaszHingeLoss(nn.Module):
    """
    Wrapper for Lovasz Hinge Loss.
    Used for the fine-tuning phase of training.
    """

    def __init__(self, per_image=True):
        super(LovaszHingeLoss, self).__init__()
        self.per_image = per_image

    def forward(self, logits, targets):
        # Lovasz hinge expects logits and binary targets
        # Squeeze channel dim if present: (B, 1, H, W) -> (B, H, W)
        if logits.dim() > 3:
            logits = logits.squeeze(1)
        if targets.dim() > 3:
            targets = targets.squeeze(1)

        return lovasz_hinge(logits, targets, per_image=self.per_image)


class DeepSupervisionLoss(nn.Module):
    """
    Wrapper to apply a base loss function to multiple outputs from the model.
    Used for U-Net++ Deep Supervision where the model returns a list of tensors.
    """

    def __init__(self, base_loss):
        super(DeepSupervisionLoss, self).__init__()
        self.base_loss = base_loss

    def forward(self, preds, targets):
        """
        preds: Tensor or list of Tensors.
        targets: Tensor (Ground Truth).
        """
        if isinstance(preds, (list, tuple)):
            loss = 0
            for pred in preds:
                loss += self.base_loss(pred, targets)
            # Average the loss over all supervision layers
            return loss / len(preds)
        else:
            return self.base_loss(preds, targets)
