import torch
import torch.nn as nn
import torch.nn.functional as F


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Jaccard extension w.r.t sorted errors
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


class LovaszHingeLoss(nn.Module):
    """
    Lovasz-Hinge loss for binary segmentation.
    Optimizes the Jaccard index (IoU) directly.

    Implements Per-Image Loss Aggregation:
    Calculates loss for each image in the batch independently and averages them.
    """

    def __init__(self):
        super(LovaszHingeLoss, self).__init__()

    def forward(self, logits, targets):
        """
        Args:
            logits: (N, C, H, W) or (N, H, W) - Raw logits from the model.
            targets: (N, H, W) - Binary ground truth masks (0 or 1).

        Returns:
            loss: Scalar tensor.
        """
        # Squeeze channel dim if present (N, 1, H, W) -> (N, H, W)
        if logits.dim() > 3:
            logits = logits.squeeze(1)

        # Ensure targets match spatial shape
        if targets.dim() > 3:
            targets = targets.squeeze(1)

        batch_size = logits.size(0)
        loss = 0.0

        # Per-Image Loss Aggregation
        # We loop over the batch to ensure the Lovasz extension is calculated
        # per image context, not over the flattened batch.
        for i in range(batch_size):
            logit_flat = logits[i].view(-1)
            target_flat = targets[i].view(-1)
            loss += lovasz_hinge_flat(logit_flat, target_flat)

        return loss / batch_size


class CombinedLoss(nn.Module):
    """
    Combines BCEWithLogitsLoss and LovaszHingeLoss.
    Used for the stabilized training pipeline.
    """

    def __init__(self, bce_weight=1.0, lovasz_weight=1.0):
        super(CombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.lovasz_weight = lovasz_weight
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.lovasz_loss = LovaszHingeLoss()

    def forward(self, logits, targets):
        """
        Args:
            logits: (N, 1, H, W) or (N, H, W)
            targets: (N, H, W)
        """
        # Prepare inputs for BCE (needs N, 1, H, W or matching logits)
        if logits.dim() == 3:
            logits_bce = logits.unsqueeze(1)
        else:
            logits_bce = logits

        if targets.dim() == 3:
            targets_bce = targets.unsqueeze(1).float()
        else:
            targets_bce = targets.float()

        bce = self.bce_loss(logits_bce, targets_bce)

        # Lovasz handles its own squeezing/flattening
        lovasz = self.lovasz_loss(logits, targets)

        return self.bce_weight * bce + self.lovasz_weight * lovasz
