import torch
import torch.nn as nn
import torch.nn.functional as F


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
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


def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] or [B, 1, H, W] Tensor, logits at each pixel
      labels: [B, H, W] or [B, 1, H, W] Tensor, binary ground truth labels (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        loss = 0
        # Iterate over batch dimension
        for input_tensor, target_tensor in zip(logits, labels):
            input_flat = input_tensor.flatten()
            target_flat = target_tensor.flatten()
            if ignore is not None:
                valid = target_flat != ignore
                input_flat = input_flat[valid]
                target_flat = target_flat[valid]
            loss = loss + lovasz_hinge_flat(input_flat, target_flat)
        return loss / logits.size(0)
    else:
        logits_flat = logits.flatten()
        labels_flat = labels.flatten()
        if ignore is not None:
            valid = labels_flat != ignore
            logits_flat = logits_flat[valid]
            labels_flat = labels_flat[valid]
        return lovasz_hinge_flat(logits_flat, labels_flat)


class LovaszHingeLoss(nn.Module):
    """
    Wrapper for Lovasz Hinge Loss
    """

    def __init__(self, per_image=True, ignore_index=None):
        super().__init__()
        self.per_image = per_image
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        return lovasz_hinge(
            logits, targets, per_image=self.per_image, ignore=self.ignore_index
        )


class BCEWithLovaszLoss(nn.Module):
    """
    Composite loss: Binary Cross Entropy + Lovasz Hinge Loss
    """

    def __init__(self, pos_weight=None, per_image=True, ignore_index=None):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.lovasz = LovaszHingeLoss(per_image=per_image, ignore_index=ignore_index)

    def forward(self, logits, targets):
        # BCEWithLogitsLoss expects float targets
        # Ensure targets are float for BCE
        bce_loss = self.bce(logits, targets.float())
        lovasz_loss = self.lovasz(logits, targets)

        # Summing maintains gradient magnitude
        return bce_loss + lovasz_loss
