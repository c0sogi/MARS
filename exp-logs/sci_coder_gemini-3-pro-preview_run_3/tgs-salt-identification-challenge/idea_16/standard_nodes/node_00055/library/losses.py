import torch
import torch.nn as nn
import torch.nn.functional as F


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
      logits: [P] Logits
      labels: [P] Labels, 0 or 1
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


def lovasz_hinge(logits, labels, per_image=True):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] or [B, 1, H, W]
      labels: [B, H, W] or [B, 1, H, W]
      per_image: compute the loss per image instead of per batch
    """
    if per_image:
        loss = 0
        for input, target in zip(logits, labels):
            input = input.view(-1)
            target = target.view(-1)
            loss += lovasz_hinge_flat(input, target)
        return loss / logits.size(0)
    else:
        return lovasz_hinge_flat(logits.view(-1), labels.view(-1))


class LovaszHingeLoss(nn.Module):
    """
    Lovasz-Hinge loss for binary segmentation.
    Optimizes the Jaccard index (IoU) directly.
    """

    def __init__(self, per_image=True):
        super().__init__()
        self.per_image = per_image

    def forward(self, logits, targets):
        # Ensure targets match logits shape (B, 1, H, W)
        if logits.dim() == 4 and targets.dim() == 3:
            targets = targets.unsqueeze(1)

        return lovasz_hinge(logits, targets, per_image=self.per_image)


class BCEDiceLoss(nn.Module):
    """
    Combination of Binary Cross Entropy and Dice Loss.
    """

    def __init__(self, bce_weight=1.0, dice_weight=1.0, smooth=1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

    def forward(self, logits, targets):
        # Ensure targets match logits shape for BCE
        if logits.dim() == 4 and targets.dim() == 3:
            targets = targets.unsqueeze(1)

        # BCE Loss (Pixel-wise)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets.float())

        # Dice Loss (Per image)
        probs = torch.sigmoid(logits)
        batch_size = probs.size(0)

        # Flatten per image
        probs = probs.view(batch_size, -1)
        targets = targets.view(batch_size, -1)

        intersection = (probs * targets).sum(1)
        union = probs.sum(1) + targets.sum(1)

        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice_score.mean()

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class DeepSupervisionLoss(nn.Module):
    """
    Wrapper to apply a base loss to multiple outputs (Deep Supervision).
    Sums the loss over all outputs.
    """

    def __init__(self, base_loss):
        super().__init__()
        self.base_loss = base_loss

    def forward(self, inputs, targets):
        # If inputs is a list/tuple, sum loss over all items
        if isinstance(inputs, (list, tuple)):
            loss = 0
            for input_tensor in inputs:
                loss += self.base_loss(input_tensor, targets)
            return loss
        else:
            return self.base_loss(inputs, targets)


def get_loss(phase_name):
    """
    Factory function to return the appropriate loss based on the training phase.

    Args:
        phase_name (str): 'phase1' or 'phase2'.

    Returns:
        nn.Module: The configured loss function.
    """
    if phase_name == "phase1":
        # Phase 1: Structural Warm-up
        # Loss: BCE + Dice
        # Scope: Deep Supervision Active (Sum over all decoder nodes)
        return DeepSupervisionLoss(BCEDiceLoss(bce_weight=1.0, dice_weight=1.0))

    elif phase_name == "phase2":
        # Phase 2: Metric Fine-tuning
        # Loss: Lovasz-Hinge (calculated per-image)
        # Scope: Deep Supervision Disabled (Loss calculated only on final output)
        # Note: The model should be configured to return a single tensor or the training loop
        # should pass only the final output. This loss expects a single tensor.
        return LovaszHingeLoss(per_image=True)

    else:
        raise ValueError(f"Unknown phase: {phase_name}")
