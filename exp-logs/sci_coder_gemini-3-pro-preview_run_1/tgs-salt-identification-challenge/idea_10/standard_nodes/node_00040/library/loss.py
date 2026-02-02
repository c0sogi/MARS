import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# Lovasz-Hinge Loss Implementation
# -----------------------------------------------------------------------------


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Jaccard loss w.r.t the sorted error
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted.float()).cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Tensor, logits at each pixel (between -\infty and +\infty)
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


def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Tensor, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth labels (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        loss = 0
        batch_size = logits.size(0)
        for i in range(batch_size):
            # Flatten: (H, W) -> (P,)
            logit_flat = logits[i].view(-1)
            label_flat = labels[i].view(-1)
            if ignore is not None:
                valid = label_flat != ignore
                logit_flat = logit_flat[valid]
                label_flat = label_flat[valid]
            loss += lovasz_hinge_flat(logit_flat, label_flat)
        return loss / batch_size
    else:
        # Flatten batch: (B, H, W) -> (B*H*W,)
        logit_flat = logits.view(-1)
        label_flat = labels.view(-1)
        if ignore is not None:
            valid = label_flat != ignore
            logit_flat = logit_flat[valid]
            label_flat = label_flat[valid]
        return lovasz_hinge_flat(logit_flat, label_flat)


class LovaszHingeLoss(nn.Module):
    def __init__(self, per_image=True, ignore=None):
        super(LovaszHingeLoss, self).__init__()
        self.per_image = per_image
        self.ignore = ignore

    def forward(self, logits, targets):
        """
        logits: (B, 1, H, W) or (B, H, W)
        targets: (B, 1, H, W) or (B, H, W)
        """
        # Squeeze channel dim if present
        if logits.dim() == 4 and logits.size(1) == 1:
            logits = logits.squeeze(1)
        if targets.dim() == 4 and targets.size(1) == 1:
            targets = targets.squeeze(1)

        return lovasz_hinge(
            logits, targets, per_image=self.per_image, ignore=self.ignore
        )


# -----------------------------------------------------------------------------
# Dice Loss Implementation
# -----------------------------------------------------------------------------


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        logits: (B, 1, H, W)
        targets: (B, 1, H, W)
        """
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits)

        # Flatten spatial dimensions: (B, -1)
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)

        intersection = (probs_flat * targets_flat).sum(1)
        union = probs_flat.sum(1) + targets_flat.sum(1)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Loss is 1 - dice
        return 1.0 - dice.mean()


# -----------------------------------------------------------------------------
# Compound Loss Implementation
# -----------------------------------------------------------------------------


class CompoundLoss(nn.Module):
    def __init__(self):
        super(CompoundLoss, self).__init__()
        self.bce_seg = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.lovasz = LovaszHingeLoss()

    def forward(self, outputs, targets):
        """
        outputs: Dictionary containing:
            - 'logits': Segmentation logits (B, 1, H, W)
        targets: Ground truth masks (B, 1, H, W) or (B, H, W)
        """
        seg_logits = outputs["logits"]

        # Ensure targets are (B, 1, H, W) float for BCE/Dice
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        targets = targets.float()

        # 1. Segmentation Losses
        # BCE
        l_bce = self.bce_seg(seg_logits, targets)

        # Dice (expects logits, applies sigmoid internally)
        l_dice = self.dice(seg_logits, targets)

        # Lovasz (expects logits, handles flattening internally)
        l_lovasz = self.lovasz(seg_logits, targets)

        # Combined Segmentation Loss
        l_seg = l_bce + l_dice + 0.1 * l_lovasz

        return l_seg
