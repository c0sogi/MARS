import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# ---------------------------------------------------------------------------
# Lovasz-Hinge Loss Implementation
# Adapted from: https://github.com/bermanmaxim/LovaszSoftmax
# ---------------------------------------------------------------------------


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


class LovaszHingeLoss(nn.Module):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """

    def __init__(self, per_image=True, ignore=None):
        super(LovaszHingeLoss, self).__init__()
        self.per_image = per_image
        self.ignore = ignore

    def forward(self, logits, labels):
        if self.per_image:
            loss = self.lovasz_hinge_flat_per_image(logits, labels)
        else:
            loss = self.lovasz_hinge_flat(logits, labels)
        return loss

    def lovasz_hinge_flat(self, logits, labels):
        """
        Binary Lovasz hinge loss
          logits: [P] Variable, logits at each pixel (between -\infty and +\infty)
          labels: [P] Tensor, binary ground truth masks (0 or 1)
          ignore: label to ignore
        """
        if self.ignore is not None:
            valid = labels != self.ignore
            logits = logits[valid]
            labels = labels[valid]

        if len(labels) == 0:
            # only void pixels, the gradients should be 0
            return logits.sum() * 0.0

        logits = logits.view(-1)
        labels = labels.view(-1)

        signs = 2.0 * labels.float() - 1.0
        errors = 1.0 - logits * Variable(signs)
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        perm = perm.data
        gt_sorted = labels[perm]
        grad = lovasz_grad(gt_sorted)
        loss = torch.dot(F.relu(errors_sorted), Variable(grad))
        return loss

    def lovasz_hinge_flat_per_image(self, logits, labels):
        """
        Binary Lovasz hinge loss computed per image and averaged
        """
        # Ensure inputs are (B, H, W) or (B, 1, H, W)
        if logits.dim() == 4:
            logits = logits.squeeze(1)
        if labels.dim() == 4:
            labels = labels.squeeze(1)

        batch_size = logits.size(0)
        losses = []
        for i in range(batch_size):
            loss = self.lovasz_hinge_flat(logits[i], labels[i])
            losses.append(loss)
        return sum(losses) / batch_size


# ---------------------------------------------------------------------------
# Dice Loss Implementation
# ---------------------------------------------------------------------------


class DiceLoss(nn.Module):
    """
    Sample-wise Dice Loss.
    Calculates Dice score for each image in the batch and averages the loss.
    """

    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        logits: (B, C, H, W) or (B, H, W) - raw output of the model (no sigmoid)
        targets: (B, C, H, W) or (B, H, W) - ground truth masks (0 or 1)
        """
        # Apply sigmoid to convert logits to probabilities
        probs = torch.sigmoid(logits)

        # Flatten label and prediction tensors
        if probs.dim() == 4:
            probs = probs.view(probs.size(0), -1)
            targets = targets.view(targets.size(0), -1)
        elif probs.dim() == 3:
            probs = probs.view(probs.size(0), -1)
            targets = targets.view(targets.size(0), -1)

        # Calculate intersection and union per sample
        intersection = (probs * targets).sum(dim=1)
        union = probs.sum(dim=1) + targets.sum(dim=1)

        # Calculate Dice score per sample
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Loss is 1 - Dice
        return 1.0 - dice_score.mean()


# ---------------------------------------------------------------------------
# Compound Loss Implementation
# ---------------------------------------------------------------------------


class CompoundLoss(nn.Module):
    """
    Consistent Compound Loss:
    L = L_BCE + L_Dice_Sample + 0.1 * L_Lovasz
    """

    def __init__(self):
        super(CompoundLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=1.0)
        self.lovasz = LovaszHingeLoss(per_image=True)

    def forward(self, logits, targets):
        """
        logits: (B, 1, H, W) or (B, H, W)
        targets: (B, 1, H, W) or (B, H, W)
        """
        # Ensure targets are float for BCE and Dice
        targets = targets.float()

        # BCE Loss (Pixel-wise)
        # BCEWithLogitsLoss handles sigmoid internally
        loss_bce = self.bce(logits, targets)

        # Dice Loss (Structural/Overlap)
        loss_dice = self.dice(logits, targets)

        # Lovasz Loss (Direct IoU Optimization)
        # Lovasz expects logits, not probabilities
        loss_lovasz = self.lovasz(logits, targets)

        # Weighted Sum
        total_loss = loss_bce + loss_dice + 0.1 * loss_lovasz

        return total_loss
