import torch
import torch.nn as nn
import torch.nn.functional as F


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Jaccard loss with respect to the sorted errors.
    See Berman et al. "The Lovasz-Softmax loss: A tractable surrogate for the optimization
    of the intersection-over-union measure in neural networks".
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union

    # Handle the case where union is 0 (though usually handled by epsilon or logic elsewhere,
    # here we assume valid masks or handle nan later if needed, but standard logic follows:)
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0 : p - 1]
    return jaccard


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss on flattened tensors.
    Args:
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


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case).
    Remove labels equal to 'ignore'.
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = labels != ignore
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels


def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss.
    Args:
        logits: [B, H, W] Tensor, logits at each pixel (between -\infty and +\infty)
        labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
        per_image: compute the loss per image instead of per batch
        ignore: void class id
    """
    if per_image:
        loss = 0
        # Iterate over batch dimension
        for input, target in zip(logits, labels):
            loss = loss + lovasz_hinge_flat(
                *flatten_binary_scores(input.unsqueeze(0), target.unsqueeze(0), ignore)
            )
        return loss / logits.size(0)
    else:
        return lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))


class SaltLoss(nn.Module):
    """
    Combined loss for Salt Segmentation: BCEWithLogits + LovaszHinge.

    This loss function combines the stability of Binary Cross Entropy with the
    direct IoU optimization of the Lovasz-Hinge loss.
    """

    def __init__(self, bce_weight=1.0, lovasz_weight=1.0):
        """
        Args:
            bce_weight (float): Weight for the BCE component.
            lovasz_weight (float): Weight for the Lovasz component.
        """
        super(SaltLoss, self).__init__()
        self.bce_weight = bce_weight
        self.lovasz_weight = lovasz_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, 1, H, W) or (B, H, W) float tensor.
            targets: (B, 1, H, W) or (B, H, W) float/int tensor.

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Align shapes for BCE
        # BCE expects (B, 1, H, W) if logits are (B, 1, H, W)
        bce_targets = targets.float()
        if logits.dim() == 4 and targets.dim() == 3:
            bce_targets = bce_targets.unsqueeze(1)

        bce_loss = self.bce(logits, bce_targets)

        # Align shapes for Lovasz
        # Lovasz logic iterates over batch, expects (B, H, W)
        lov_logits = logits
        lov_targets = targets

        if lov_logits.dim() == 4:
            lov_logits = lov_logits.squeeze(1)
        if lov_targets.dim() == 4:
            lov_targets = lov_targets.squeeze(1)

        lov_loss = lovasz_hinge(lov_logits, lov_targets, per_image=True)

        return self.bce_weight * bce_loss + self.lovasz_weight * lov_loss
