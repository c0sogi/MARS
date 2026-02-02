import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------------------------------------------------
# Lovasz-Hinge Loss Helpers
# -------------------------------------------------------------------------


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


# -------------------------------------------------------------------------
# Loss Modules
# -------------------------------------------------------------------------


class SampleWiseDiceLoss(nn.Module):
    """
    Calculates Dice coefficient per image and averages it.
    Handles empty masks correctly (IoU/Dice of empty pred and empty truth is 1).
    """

    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        logits: (N, C, H, W) or (N, H, W) - Raw scores (before sigmoid)
        targets: (N, C, H, W) or (N, H, W) - 0 or 1
        """
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits)

        # Flatten spatial dimensions: (N, ...) -> (N, -1)
        if probs.dim() > 2:
            probs = probs.view(probs.size(0), -1)

        if targets.dim() > 2:
            targets = targets.view(targets.size(0), -1)

        # Calculate intersection and union per sample
        intersection = (probs * targets).sum(dim=1)
        union = probs.sum(dim=1) + targets.sum(dim=1)

        # Dice coefficient per sample
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Loss is 1 - Mean Dice
        return 1.0 - dice.mean()


class LovaszHingeLoss(nn.Module):
    """
    Lovasz-Hinge loss for binary segmentation.
    Optimizes the Jaccard index (IoU) using the Lovasz extension.
    """

    def __init__(self):
        super().__init__()

    def forward(self, logits, targets):
        """
        logits: (N, 1, H, W) or (N, H, W)
        targets: (N, 1, H, W) or (N, H, W)
        """
        # Squeeze channel dim if present
        if logits.dim() > 3:
            logits = logits.squeeze(1)
        if targets.dim() > 3:
            targets = targets.squeeze(1)

        # Flatten everything to 1D vectors
        logits_flat = logits.view(-1)
        targets_flat = targets.view(-1)

        return lovasz_hinge_flat(logits_flat, targets_flat)


class BCEDiceLoss(nn.Module):
    """
    Weighted combination of Binary Cross Entropy and Sample-Wise Dice Loss.
    Used for Phase 1 of training (Robust Convergence).
    """

    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = SampleWiseDiceLoss()

    def forward(self, logits, targets):
        # Ensure targets match logits shape for BCE
        if targets.shape != logits.shape:
            # If targets is (N, H, W) and logits is (N, 1, H, W)
            if targets.dim() == logits.dim() - 1:
                targets = targets.unsqueeze(1)

        loss_bce = self.bce(logits, targets.float())
        loss_dice = self.dice(logits, targets)

        return self.bce_weight * loss_bce + self.dice_weight * loss_dice


class BCELovaszLoss(nn.Module):
    """
    Weighted combination of Binary Cross Entropy and Lovasz-Hinge Loss.
    Used for Phase 2 of training (Metric Fine-tuning).
    """

    def __init__(self, bce_weight=0.5, lovasz_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.lovasz_weight = lovasz_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.lovasz = LovaszHingeLoss()

    def forward(self, logits, targets):
        # Ensure targets match logits shape for BCE
        if targets.shape != logits.shape:
            if targets.dim() == logits.dim() - 1:
                targets = targets.unsqueeze(1)

        loss_bce = self.bce(logits, targets.float())
        loss_lovasz = self.lovasz(logits, targets)

        return self.bce_weight * loss_bce + self.lovasz_weight * loss_lovasz


class DeepSupervisionLoss(nn.Module):
    """
    Wrapper to apply a loss function to multiple outputs from the model.
    Used when the model returns a list of tensors (Main Output + Aux Outputs).
    """

    def __init__(self, criterion):
        super().__init__()
        self.criterion = criterion

    def forward(self, outputs, targets):
        # Check if outputs is a list or tuple (Deep Supervision enabled)
        if isinstance(outputs, (list, tuple)):
            loss = 0
            num_outputs = len(outputs)

            # Apply criterion to each output level
            # Note: Assuming all outputs are upsampled to match target resolution
            for output in outputs:
                loss += self.criterion(output, targets)

            # Average the loss across all heads to maintain magnitude consistency
            return loss / num_outputs
        else:
            # Standard single output
            return self.criterion(outputs, targets)
