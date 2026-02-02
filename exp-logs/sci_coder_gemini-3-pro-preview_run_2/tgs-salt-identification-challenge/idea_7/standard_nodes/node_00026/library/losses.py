import torch
import torch.nn as nn
import torch.nn.functional as F
from library import config


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Jaccard loss w.r.t the sorted error.
    See Berman et al., "The Lovasz-Softmax loss: A tractable surrogate for the optimization of the intersection-over-union measure in neural networks".
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
    Binary Lovasz hinge loss on flattened tensors.

    Args:
        logits: [P] Float, logits of the prediction (before sigmoid)
        labels: [P] Float, binary labels (0 or 1)
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.0

    signs = 2.0 * labels - 1.0
    errors = 1.0 - logits * signs
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), grad)
    return loss


class LovaszHingeLoss(nn.Module):
    """
    Binary Lovasz Hinge Loss for segmentation.
    """

    def __init__(self):
        super(LovaszHingeLoss, self).__init__()

    def forward(self, logits, targets):
        """
        Args:
            logits: [B, 1, H, W] or [B, H, W]
            targets: [B, H, W] or [B, 1, H, W]
        """
        # Flatten predictions and targets
        logits_flat = logits.view(-1)
        targets_flat = targets.view(-1)

        return lovasz_hinge_flat(logits_flat, targets_flat)


class SaltNetLoss(nn.Module):
    """
    Composite Loss for Salt Segmentation.

    L_total = L_bce + L_lovasz
    """

    def __init__(self):
        super(SaltNetLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.lovasz = LovaszHingeLoss()

    def forward(self, seg_logits, seg_targets):
        """
        Args:
            seg_logits: (Tensor) Segmentation logits [B, 1, H, W]
            seg_targets: (Tensor) Ground truth masks [B, 1, H, W] or [B, H, W]

        Returns:
            loss: (Tensor) Scalar loss value
            metrics: (Dict) Dictionary containing individual loss components for logging
        """
        # --- Segmentation Loss ---
        # Ensure targets are float for BCE
        if seg_targets.dtype != torch.float32:
            seg_targets = seg_targets.float()

        # Ensure shape alignment for BCE
        if seg_targets.ndim == 3:
            seg_targets = seg_targets.unsqueeze(1)

        loss_bce = self.bce(seg_logits, seg_targets)
        loss_lovasz = self.lovasz(seg_logits, seg_targets)

        total_loss = loss_bce + loss_lovasz

        metrics = {
            "loss_bce": loss_bce.item(),
            "loss_lovasz": loss_lovasz.item(),
            "loss_total": total_loss.item(),
        }

        return total_loss, metrics
