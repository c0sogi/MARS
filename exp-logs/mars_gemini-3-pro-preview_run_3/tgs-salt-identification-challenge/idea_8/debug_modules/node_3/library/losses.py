import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEDiceLoss(nn.Module):
    """
    Weighted combination of Binary Cross Entropy and Dice Loss.
    Used for the warm-up phase of training to establish convergence.
    """

    def __init__(self, bce_weight=0.5, smooth=1.0):
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = 1.0 - bce_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        """
        Args:
            inputs: (B, 1, H, W) or (B, H, W) logits
            targets: (B, H, W) or (B, 1, H, W) binary masks (0 or 1)
        """
        # Ensure inputs and targets are flattened and float
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1).float()

        # Binary Cross Entropy
        bce_loss = self.bce(inputs_flat, targets_flat)

        # Dice Loss
        pred = torch.sigmoid(inputs_flat)
        intersection = (pred * targets_flat).sum()
        union = pred.sum() + targets_flat.sum()

        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice_score

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class LovaszHingeLoss(nn.Module):
    """
    Lovasz-Hinge loss for binary segmentation.
    Optimizes the Jaccard index (IoU) directly using the Lovasz extension.
    Calculates loss per-image and averages over the batch to align with the
    Mean Average Precision metric.
    """

    def __init__(self):
        super(LovaszHingeLoss, self).__init__()

    def forward(self, inputs, targets):
        """
        Args:
            inputs: (B, 1, H, W) logits
            targets: (B, H, W) or (B, 1, H, W) binary masks
        """
        # Squeeze channel dimension if present to get (B, H, W)
        if inputs.dim() == 4:
            inputs = inputs.squeeze(1)
        if targets.dim() == 4:
            targets = targets.squeeze(1)

        # Ensure targets are float
        targets = targets.float()

        batch_size = inputs.size(0)
        total_loss = 0.0

        for i in range(batch_size):
            # Calculate loss for each image individually
            input_flat = inputs[i].view(-1)
            target_flat = targets[i].view(-1)

            loss = self._lovasz_hinge_flat(input_flat, target_flat)
            total_loss += loss

        return total_loss / batch_size

    def _lovasz_hinge_flat(self, logits, labels):
        """
        Binary Lovasz hinge loss for a flat vector.
        Args:
            logits: [P] Logits at each pixel (between -inf and +inf)
            labels: [P] Binary ground truth labels (0 or 1)
        """
        if len(labels) == 0:
            # Should not happen with valid images
            return logits.sum() * 0.0

        signs = 2.0 * labels - 1.0
        errors = 1.0 - logits * signs
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        perm = perm.data

        gt_sorted = labels[perm]
        grad = self._lovasz_grad(gt_sorted)

        loss = torch.dot(F.relu(errors_sorted), grad)
        return loss

    def _lovasz_grad(self, gt_sorted):
        """
        Computes gradient of the Lovasz extension w.r.t sorted errors.
        """
        p = len(gt_sorted)
        gts = gt_sorted.sum()

        # Intersection: gts - cumulative sum of gt_sorted
        intersection = gts - gt_sorted.float().cumsum(0)

        # Union: gts + cumulative sum of (1 - gt_sorted)
        union = gts + (1.0 - gt_sorted).float().cumsum(0)

        jaccard = 1.0 - intersection / union

        if p > 1:
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]

        return jaccard
