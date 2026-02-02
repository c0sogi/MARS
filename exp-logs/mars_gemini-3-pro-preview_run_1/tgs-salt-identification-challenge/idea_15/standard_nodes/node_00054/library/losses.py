import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Sample-wise Dice Loss for binary segmentation.
    Calculates Dice coefficient per image and averages over the batch.
    """

    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, 1, H, W) or (B, H, W) raw logits
            targets: (B, 1, H, W) or (B, H, W) binary targets (0 or 1)
        """
        # Apply sigmoid to logits
        probs = torch.sigmoid(logits)

        # Flatten label and prediction tensors
        # Keep batch dimension: (B, -1)
        if probs.dim() > 2:
            probs = probs.view(probs.size(0), -1)

        if targets.dim() > 2:
            targets = targets.view(targets.size(0), -1)

        # Ensure targets are float
        targets = targets.float()

        # Calculate intersection and union per sample
        intersection = (probs * targets).sum(dim=1)
        union = probs.sum(dim=1) + targets.sum(dim=1)

        # Calculate Dice score
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Return mean loss (1 - dice)
        return 1.0 - dice.mean()


class LovaszHingeLoss(nn.Module):
    """
    Lovasz-Hinge Loss for binary segmentation.
    Optimizes the Jaccard index (IoU) directly using the Lovasz extension.
    """

    def __init__(self, per_image=True, ignore_index=None):
        super(LovaszHingeLoss, self).__init__()
        self.per_image = per_image
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, 1, H, W) or (B, H, W) raw logits
            targets: (B, 1, H, W) or (B, H, W) binary targets (0 or 1)
        """
        if self.per_image:
            batch_size = logits.size(0)
            loss = 0
            for i in range(batch_size):
                # Process each image in the batch
                logit_flat = logits[i].view(-1)
                target_flat = targets[i].view(-1)
                loss += self._lovasz_hinge_flat(logit_flat, target_flat)
            return loss / batch_size
        else:
            # Process whole batch at once
            return self._lovasz_hinge_flat(logits.view(-1), targets.view(-1))

    def _lovasz_hinge_flat(self, logits, labels):
        """
        Binary Lovasz hinge loss on flattened tensors.
        """
        if self.ignore_index is not None:
            mask = labels != self.ignore_index
            logits = logits[mask]
            labels = labels[mask]

        # Treat 0 as -1 for hinge loss logic if needed, but standard implementation
        # uses errors = 1 - logits * signs.
        # Here we assume labels are 0/1.
        # If label is 1, error is relu(1 - logit)
        # If label is 0, error is relu(1 + logit)
        # This is equivalent to: signs = 2*labels - 1; errors = relu(1 - logits * signs)

        signs = 2.0 * labels.float() - 1.0
        errors = F.relu(1.0 - logits * signs)

        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        perm = perm.data

        gt_sorted = labels.float()[perm]
        grad = self._lovasz_grad(gt_sorted)

        loss = torch.dot(errors_sorted, grad)
        return loss

    def _lovasz_grad(self, gt_sorted):
        """
        Computes gradient of the Lovasz extension w.r.t sorted errors
        """
        p = len(gt_sorted)
        gts = gt_sorted.sum()

        intersection = gts - gt_sorted.cumsum(0)
        union = gts + (1 - gt_sorted).cumsum(0)

        jaccard = 1.0 - intersection / union

        if p > 1:  # cover 1-pixel case
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]

        return jaccard


class BCEDiceLoss(nn.Module):
    """
    Combination of BCEWithLogitsLoss and DiceLoss.
    Used for Phase 1 of the curriculum.
    """

    def __init__(self, bce_weight=1.0, dice_weight=1.0, smooth=1.0):
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, logits, targets):
        # Ensure targets match logits shape/type for BCE
        # BCEWithLogitsLoss expects float targets
        targets_float = targets.float()

        # If logits is (B, 1, H, W) and targets is (B, H, W), unsqueeze targets
        if logits.dim() == 4 and targets_float.dim() == 3:
            targets_float = targets_float.unsqueeze(1)

        bce_loss = self.bce(logits, targets_float)
        dice_loss = self.dice(logits, targets)

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class BCELovaszLoss(nn.Module):
    """
    Combination of BCEWithLogitsLoss and LovaszHingeLoss.
    Used for Phase 2 of the curriculum (Substitutive Loss).
    """

    def __init__(self, bce_weight=1.0, lovasz_weight=1.0, per_image=True):
        super(BCELovaszLoss, self).__init__()
        self.bce_weight = bce_weight
        self.lovasz_weight = lovasz_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.lovasz = LovaszHingeLoss(per_image=per_image)

    def forward(self, logits, targets):
        # Ensure targets match logits shape/type for BCE
        targets_float = targets.float()

        if logits.dim() == 4 and targets_float.dim() == 3:
            targets_float = targets_float.unsqueeze(1)

        bce_loss = self.bce(logits, targets_float)
        lovasz_loss = self.lovasz(logits, targets)

        return self.bce_weight * bce_loss + self.lovasz_weight * lovasz_loss
