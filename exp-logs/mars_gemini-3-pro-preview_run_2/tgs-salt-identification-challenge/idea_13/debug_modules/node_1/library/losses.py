import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Jaccard loss with respect to the sorted errors.
    See: https://github.com/bermanmaxim/LovaszSoftmax
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
    Args:
        logits: [P] Variable, logits of the prediction
        labels: [P] Tensor, binary ground truth labels (0 or 1)
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.0

    signs = 2.0 * labels.float() - 1.0
    errors = 1.0 - logits * Variable(signs)
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), Variable(grad))
    return loss


class LovaszHingeLoss(nn.Module):
    """
    Binary Lovasz hinge loss.
    Optimizes the Jaccard index (IoU) directly using the Lovasz extension.
    """

    def __init__(self, per_image=True, ignore_index=None):
        super().__init__()
        self.per_image = per_image
        self.ignore_index = ignore_index

    def forward(self, logits, labels):
        """
        Args:
            logits: [B, 1, H, W] or [B, H, W] logits
            labels: [B, 1, H, W] or [B, H, W] binary labels (0 or 1)
        """
        # Squeeze channel dimension if present
        if logits.dim() > 3:
            logits = logits.squeeze(1)
        if labels.dim() > 3:
            labels = labels.squeeze(1)

        if self.per_image:
            loss = 0
            for input_flat, target_flat in zip(
                logits.view(logits.size(0), -1), labels.view(labels.size(0), -1)
            ):
                # Filter ignore_index if specified
                if self.ignore_index is not None:
                    mask = target_flat != self.ignore_index
                    input_flat = input_flat[mask]
                    target_flat = target_flat[mask]

                loss += lovasz_hinge_flat(input_flat, target_flat)
            return loss / logits.size(0)
        else:
            # Flatten entire batch
            logits_flat = logits.view(-1)
            labels_flat = labels.view(-1)

            if self.ignore_index is not None:
                mask = labels_flat != self.ignore_index
                logits_flat = logits_flat[mask]
                labels_flat = labels_flat[mask]

            return lovasz_hinge_flat(logits_flat, labels_flat)


class BCELovaszLoss(nn.Module):
    """
    Composite loss function: Sum of Binary Cross Entropy and Lovasz Hinge Loss.
    BCE provides strong pixel-wise gradients, while Lovasz optimizes the IoU metric directly.
    """

    def __init__(self, bce_weight=1.0, lovasz_weight=1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.lovasz_weight = lovasz_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.lovasz = LovaszHingeLoss()

    def forward(self, logits, labels):
        """
        Args:
            logits: [B, 1, H, W] or [B, H, W]
            labels: [B, 1, H, W] or [B, H, W]
        """
        # Ensure labels are float for BCE
        labels_float = labels.float()

        # Adjust shapes if necessary to match BCE expectation
        # BCEWithLogitsLoss expects same shape for input and target
        if logits.shape != labels_float.shape:
            # If logits is (B, 1, H, W) and labels is (B, H, W) or vice versa
            if logits.dim() == 4 and labels_float.dim() == 3:
                labels_float = labels_float.unsqueeze(1)
            elif logits.dim() == 3 and labels_float.dim() == 4:
                logits = logits.unsqueeze(1)

        bce_loss = self.bce(logits, labels_float)
        lovasz_loss = self.lovasz(logits, labels)

        return (self.bce_weight * bce_loss) + (self.lovasz_weight * lovasz_loss)
