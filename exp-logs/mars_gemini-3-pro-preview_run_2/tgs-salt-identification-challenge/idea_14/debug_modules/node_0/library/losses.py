import torch
import torch.nn as nn
import torch.nn.functional as F


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension of jaccard loss.

    Args:
        gt_sorted (torch.Tensor): Sorted ground truth labels.

    Returns:
        torch.Tensor: Gradient of the Jaccard loss.
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
        logits (torch.Tensor): Logits at each pixel (N,).
        labels (torch.Tensor): Binary ground truth labels (N,).

    Returns:
        torch.Tensor: The calculated loss scalar.
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
    Wrapper for Lovasz Hinge Loss to be used as a PyTorch module.
    """

    def __init__(self):
        super(LovaszHingeLoss, self).__init__()

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): Predictions of shape (N, C, H, W) or (N, H, W).
            targets (torch.Tensor): Ground truth of shape (N, C, H, W) or (N, H, W).

        Returns:
            torch.Tensor: Scalar loss.
        """
        # Flatten the tensors
        logits_flat = logits.reshape(-1)
        targets_flat = targets.reshape(-1)
        return lovasz_hinge_flat(logits_flat, targets_flat)


class CombinedLoss(nn.Module):
    """
    Combined loss function: Sum of BCEWithLogitsLoss and LovaszHingeLoss.
    Designed to stabilize training with BCE while optimizing IoU with Lovasz.
    """

    def __init__(self, bce_weight=1.0, lovasz_weight=1.0):
        super(CombinedLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.lovasz = LovaszHingeLoss()
        self.bce_weight = bce_weight
        self.lovasz_weight = lovasz_weight

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): Predictions.
            targets (torch.Tensor): Ground truth.

        Returns:
            torch.Tensor: Weighted sum of BCE and Lovasz loss.
        """
        bce_loss = self.bce(logits, targets)
        lovasz_loss = self.lovasz(logits, targets)
        return (self.bce_weight * bce_loss) + (self.lovasz_weight * lovasz_loss)
