import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os


# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


set_seed(42)


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Jaccard loss w.r.t the sorted error
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - torch.cumsum(gt_sorted, 0)
    union = gts + torch.cumsum(1 - gt_sorted, 0)
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
    Calculates loss per image and averages over the batch.
    """

    def __init__(self):
        super(LovaszHingeLoss, self).__init__()

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C, H, W) or (B, H, W) - Raw output of the network (no sigmoid)
            targets: (B, C, H, W) or (B, H, W) - Binary ground truth masks (0 or 1)
        """
        # Squeeze channel dimension if present (assuming binary segmentation C=1)
        if logits.dim() == 4:
            logits = logits.squeeze(1)
        if targets.dim() == 4:
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
    Wrapper for BCEWithLogitsLoss for numerical stability.
    """

    def __init__(self):
        super(StableBCELoss, self).__init__()
        # reduction='mean' is standard, but CombinedLoss will handle per-image reduction manually
        # to match the Lovasz aggregation strategy.
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits, targets):
        if logits.dim() == 4:
            logits = logits.squeeze(1)
        if targets.dim() == 4:
            targets = targets.squeeze(1)

        # Calculate per-pixel loss
        loss = self.bce(logits, targets.float())

        # Mean over spatial dimensions (H, W) to get per-image loss
        loss = loss.view(loss.size(0), -1).mean(dim=1)

        # Mean over batch
        return loss.mean()


class CombinedLoss(nn.Module):
    """
    Stabilized Composite Loss: Sum of Lovasz-Hinge and BCE.
    Critically, this calculates the combined error PER IMAGE and then averages across the batch.
    This prevents outliers (common in pseudo-labeling) from dominating the gradients.
    """

    def __init__(self, bce_weight=1.0, lovasz_weight=1.0):
        super(CombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.lovasz_weight = lovasz_weight
        self.bce_func = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, 1, H, W) or (B, H, W)
            targets: (B, 1, H, W) or (B, H, W)
        """
        # Standardization
        if logits.dim() == 4:
            logits = logits.squeeze(1)
        if targets.dim() == 4:
            targets = targets.squeeze(1)

        batch_size = logits.size(0)
        total_loss = 0.0

        for i in range(batch_size):
            # Flatten for this specific image
            logit_flat = logits[i].view(-1)
            target_flat = targets[i].view(-1)

            # 1. Lovasz Hinge (Per Image)
            l_lovasz = lovasz_hinge_flat(logit_flat, target_flat)

            # 2. BCE (Per Image)
            # We compute BCE on the flattened vectors for consistency
            l_bce = self.bce_func(logit_flat, target_flat.float()).mean()

            # Sum per image
            image_loss = (self.bce_weight * l_bce) + (self.lovasz_weight * l_lovasz)
            total_loss += image_loss

        # Average across batch
        return total_loss / batch_size
