import torch
import torch.nn as nn
import torch.nn.functional as F


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.cumsum(0)
    union = gts + (1 - gt_sorted).cumsum(0)
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
    gt_sorted = labels.float()[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), grad)
    return loss


class LovaszHingeLoss(nn.Module):
    """
    Lovasz-Hinge loss for binary segmentation.
    Computes the loss per-image and averages across the batch to align with
    Mean Average Precision metric and prevent gradient divergence.
    """

    def __init__(self):
        super().__init__()

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C, H, W) or (B, H, W) logits (before sigmoid).
            targets: (B, C, H, W) or (B, H, W) binary targets.
        """
        # Squeeze channel dim if present (assuming binary class C=1)
        if logits.dim() > 3:
            logits = logits.squeeze(1)
        if targets.dim() > 3:
            targets = targets.squeeze(1)

        batch_size = logits.size(0)
        losses = []

        for i in range(batch_size):
            # Flatten spatial dimensions
            logit_flat = logits[i].view(-1)
            target_flat = targets[i].view(-1)

            loss = lovasz_hinge_flat(logit_flat, target_flat)
            losses.append(loss)

        return torch.stack(losses).mean()


class StableBCELoss(nn.Module):
    """
    Binary Cross Entropy with Logits Loss.
    Numerically stable and supports soft targets for distillation.
    """

    def __init__(self):
        super().__init__()
        # reduction='mean' averages over the batch and pixels.
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C, H, W) or (B, H, W) logits.
            targets: (B, C, H, W) or (B, H, W) targets (binary or soft).
        """
        # Ensure targets are float for BCEWithLogitsLoss
        return self.bce(logits, targets.float())


class CombinedLoss(nn.Module):
    """
    Sum of Lovasz-Hinge and BCE Loss.
    Used for supervised training stages where targets are binary.
    """

    def __init__(self):
        super().__init__()
        self.lovasz = LovaszHingeLoss()
        self.bce = StableBCELoss()

    def forward(self, logits, targets):
        loss_lovasz = self.lovasz(logits, targets)
        loss_bce = self.bce(logits, targets)

        return loss_lovasz + loss_bce
