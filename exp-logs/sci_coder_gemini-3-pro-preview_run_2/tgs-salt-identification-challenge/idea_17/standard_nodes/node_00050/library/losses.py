import torch
import torch.nn as nn
import torch.nn.functional as F


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Jaccard loss w.r.t the sorted error
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
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), grad)
    return loss


class LovaszHingeLoss(nn.Module):
    """
    Lovasz Hinge Loss for Binary Segmentation.
    Optimizes the Jaccard index (IoU) directly using the Lovasz extension.
    Calculates loss per image and averages across the batch.
    """

    def __init__(self):
        super(LovaszHingeLoss, self).__init__()

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, 1, H, W) or (B, H, W) raw logits
            targets: (B, 1, H, W) or (B, H, W) binary targets (0 or 1)
        """
        # Squeeze channel dim if present
        if logits.dim() > 3:
            logits = logits.squeeze(1)
        if targets.dim() > 3:
            targets = targets.squeeze(1)

        batch_size = logits.size(0)
        loss = 0.0

        for i in range(batch_size):
            logit_flat = logits[i].view(-1)
            target_flat = targets[i].view(-1)
            loss += lovasz_hinge_flat(logit_flat, target_flat)

        return loss / batch_size


class StableBCELoss(nn.Module):
    """
    Stable Binary Cross Entropy Loss using BCEWithLogitsLoss.
    Supports both binary targets and soft targets (probabilities).
    """

    def __init__(self):
        super(StableBCELoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, 1, H, W) raw logits
            targets: (B, 1, H, W) targets (0/1 or probabilities)
        """
        return self.bce(logits, targets)


class CombinedLoss(nn.Module):
    """
    Aggregates LovaszHingeLoss and StableBCELoss.
    Handles different modes for Supervised and Semi-Supervised (Distillation) training.
    """

    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.lovasz = LovaszHingeLoss()
        self.bce = StableBCELoss()

    def forward(self, logits, targets, mode="supervised"):
        """
        Args:
            logits: (B, 1, H, W) raw network output
            targets: (B, 1, H, W) Ground truth or Soft Targets
            mode: 'supervised' or 'distillation'

        Returns:
            Calculated loss value.
        """
        # Ensure targets match logits shape
        if targets.shape != logits.shape:
            # If targets are (B, H, W), unsqueeze to (B, 1, H, W)
            if targets.dim() == 3:
                targets = targets.unsqueeze(1)

        # Cast targets to float for calculation
        targets = targets.float()

        if mode == "supervised":
            # Labeled Data: Lovasz (IoU optim) + BCE (Pixel-wise accuracy)
            # Lovasz expects binary targets, BCE can handle float but here it's binary
            lovasz_loss = self.lovasz(logits, targets)
            bce_loss = self.bce(logits, targets)
            return lovasz_loss + bce_loss

        elif mode == "distillation":
            # Unlabeled Data: Distill Teacher's soft probabilities using BCE.
            # Lovasz is not suitable for soft targets as it relies on sorting binary errors.
            return self.bce(logits, targets)

        else:
            raise ValueError(f"Unknown loss mode: {mode}")
