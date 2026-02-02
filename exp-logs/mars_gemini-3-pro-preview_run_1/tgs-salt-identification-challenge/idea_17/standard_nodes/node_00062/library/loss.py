import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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
    loss = torch.dot(F.elu(errors_sorted) + 1, grad)
    return loss


class LovaszHingeLoss(nn.Module):
    """
    Lovasz Hinge Loss for Binary Segmentation.
    Optimizes the Jaccard index directly.
    """

    def __init__(self, per_image=True):
        super().__init__()
        self.per_image = per_image

    def forward(self, logits, targets):
        """
        Args:
            logits: (N, 1, H, W) or (N, H, W)
            targets: (N, 1, H, W) or (N, H, W)
        """
        if logits.dim() > 2:
            # Flatten to (N, -1) if per_image, else (-1)
            if self.per_image:
                logits = logits.view(logits.size(0), -1)
                targets = targets.view(targets.size(0), -1)
            else:
                logits = logits.view(-1)
                targets = targets.view(-1)

        if self.per_image:
            loss = 0
            for i in range(logits.size(0)):
                loss += lovasz_hinge_flat(logits[i], targets[i])
            return loss / logits.size(0)
        else:
            return lovasz_hinge_flat(logits, targets)


class SampleWiseDiceLoss(nn.Module):
    """
    Dice Loss calculated per sample and averaged.
    Better for handling empty masks than batch-wise Dice.
    """

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits)

        # Flatten spatial dimensions: (N, C, H, W) -> (N, -1)
        # Assuming C=1 for binary segmentation
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)

        intersection = (probs_flat * targets_flat).sum(dim=1)
        union = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Return 1 - mean dice
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    """
    Combination of BCEWithLogitsLoss and SampleWiseDiceLoss.
    Used for Phase 1 (Structure) training.
    """

    def __init__(self, bce_weight=1.0, dice_weight=1.0, smooth=1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = SampleWiseDiceLoss(smooth=smooth)
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class BCELovaszLoss(nn.Module):
    """
    Combination of BCEWithLogitsLoss and LovaszHingeLoss.
    Used for Phase 2 (Metric) training.
    """

    def __init__(self, bce_weight=1.0, lovasz_weight=1.0, per_image=True):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.lovasz = LovaszHingeLoss(per_image=per_image)
        self.bce_weight = bce_weight
        self.lovasz_weight = lovasz_weight

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        lovasz_loss = self.lovasz(logits, targets)
        return self.bce_weight * bce_loss + self.lovasz_weight * lovasz_loss


class DeepSupervisionLoss(nn.Module):
    """
    Wrapper to handle Deep Supervision outputs.
    Applies main_loss to the first output and aux_loss to subsequent outputs.
    """

    def __init__(self, main_loss_fn, aux_loss_fn, aux_weights=None):
        """
        Args:
            main_loss_fn: Loss function for the final output (e.g., BCELovaszLoss).
            aux_loss_fn: Loss function for auxiliary heads (e.g., BCEDiceLoss).
            aux_weights: List of weights for aux heads. If None, defaults to 1.0.
        """
        super().__init__()
        self.main_loss_fn = main_loss_fn
        self.aux_loss_fn = aux_loss_fn
        self.aux_weights = aux_weights

    def forward(self, preds, targets):
        """
        Args:
            preds: List or Tuple of tensors [main_pred, aux1, aux2, ...]
                   OR single tensor main_pred
            targets: Ground truth tensor
        """
        # Handle single output case
        if not isinstance(preds, (list, tuple)):
            return self.main_loss_fn(preds, targets)

        # Main Head Loss
        main_pred = preds[0]
        # Ensure target shape matches if needed (though usually U-Net outputs match input)
        # If model output is different size, interpolate prediction to target size
        if main_pred.shape[2:] != targets.shape[2:]:
            main_pred = F.interpolate(
                main_pred, size=targets.shape[2:], mode="bilinear", align_corners=False
            )

        total_loss = self.main_loss_fn(main_pred, targets)

        # Auxiliary Heads Loss
        for i, aux_pred in enumerate(preds[1:]):
            weight = 1.0
            if self.aux_weights is not None and i < len(self.aux_weights):
                weight = self.aux_weights[i]

            # Interpolate aux prediction to target size if necessary
            if aux_pred.shape[2:] != targets.shape[2:]:
                aux_pred = F.interpolate(
                    aux_pred,
                    size=targets.shape[2:],
                    mode="bilinear",
                    align_corners=False,
                )

            total_loss += weight * self.aux_loss_fn(aux_pred, targets)

        return total_loss
