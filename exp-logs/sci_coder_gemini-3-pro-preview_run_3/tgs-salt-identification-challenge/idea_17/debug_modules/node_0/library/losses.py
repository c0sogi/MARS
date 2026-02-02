import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# Helper Functions for Lovasz-Hinge Loss
# =============================================================================


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Jaccard loss w.r.t the sorted error.
    See Alg. 1 in paper: https://arxiv.org/abs/1705.08790
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
    Binary Lovasz hinge loss for a flat vector.
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


def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss implementation.
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth labels (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class label
    """
    if per_image:
        loss = 0
        for input, target in zip(logits, labels):
            input = input.view(-1)
            target = target.view(-1)
            if ignore is not None:
                valid = target != ignore
                input = input[valid]
                target = target[valid]
            loss = loss + lovasz_hinge_flat(input, target)
        return loss / logits.size(0)
    else:
        logits = logits.view(-1)
        labels = labels.view(-1)
        if ignore is not None:
            valid = labels != ignore
            logits = logits[valid]
            labels = labels[valid]
        return lovasz_hinge_flat(logits, labels)


# =============================================================================
# Loss Modules
# =============================================================================


class LovaszHingeLoss(nn.Module):
    """
    Lovasz-Hinge loss for binary segmentation.
    Optimizes the Jaccard index (IoU) directly.
    """

    def __init__(self, per_image=True, ignore=None):
        super().__init__()
        self.per_image = per_image
        self.ignore = ignore

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, 1, H, W) or (B, H, W)
            targets: (B, 1, H, W) or (B, H, W)
        """
        # Squeeze channel dimension if present to match (B, H, W) expected by lovasz_hinge
        if logits.dim() > 3:
            logits = logits.squeeze(1)
        if targets.dim() > 3:
            targets = targets.squeeze(1)

        return lovasz_hinge(
            logits, targets, per_image=self.per_image, ignore=self.ignore
        )


class DiceLoss(nn.Module):
    """
    Dice Coefficient Loss.
    """

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # Apply sigmoid to convert logits to probabilities
        probs = torch.sigmoid(logits)

        # Flatten tensors
        probs = probs.view(-1)
        targets = targets.view(-1)

        intersection = (probs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (
            probs.sum() + targets.sum() + self.smooth
        )

        return 1 - dice


class BCEDiceLoss(nn.Module):
    """
    Combination of Binary Cross Entropy and Dice Loss.
    Used for structural warmup (Phase 1).
    """

    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, logits, targets):
        # Ensure targets are float for BCE
        if targets.dtype != torch.float32:
            targets = targets.float()

        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class DeepSupervisionLoss(nn.Module):
    """
    Wrapper for Deep Supervision.
    Applies the base loss to a list of model outputs with specified weights.
    """

    def __init__(self, base_loss, weights=None):
        """
        Args:
            base_loss: The loss module to apply (e.g., BCEDiceLoss or LovaszHingeLoss).
            weights: List of float weights for each output head.
                     If None, assumes uniform weighting (1.0) for all heads.
        """
        super().__init__()
        self.base_loss = base_loss
        self.weights = weights

    def forward(self, preds, targets):
        """
        Args:
            preds: Tensor or List[Tensor]. Model outputs.
            targets: Tensor. Ground truth masks.
        """
        # Handle single output case
        if not isinstance(preds, (list, tuple)):
            return self.base_loss(preds, targets)

        # Handle deep supervision (list of outputs)
        loss = 0.0
        num_heads = len(preds)

        # Determine weights
        if self.weights is None:
            current_weights = [1.0] * num_heads
        else:
            current_weights = self.weights
            # Handle mismatch length if necessary, though config should be correct
            if len(current_weights) < num_heads:
                # Pad with 0 or last weight? Safer to assume 0 for extra heads not specified
                current_weights = current_weights + [0.0] * (
                    num_heads - len(current_weights)
                )

        total_active_weight = 0.0

        for i, pred in enumerate(preds):
            w = current_weights[i]
            if w > 0:
                # Check spatial dimensions
                if pred.shape[-2:] != targets.shape[-2:]:
                    # Interpolate targets to match prediction size (nearest neighbor for masks)
                    target_res = F.interpolate(
                        targets, size=pred.shape[-2:], mode="nearest"
                    )
                else:
                    target_res = targets

                loss += w * self.base_loss(pred, target_res)
                total_active_weight += w

        # Return aggregated loss
        return loss
