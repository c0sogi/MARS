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
    union = gts + (1 - gt_sorted.float()).cumsum(0)
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
    loss = torch.dot(F.relu(errors_sorted), grad)
    return loss


class LovaszHingeLoss(nn.Module):
    """
    Lovasz Hinge Loss for binary segmentation.
    """

    def __init__(self, per_image=True):
        super(LovaszHingeLoss, self).__init__()
        self.per_image = per_image

    def forward(self, logits, targets):
        """
        Args:
            logits: (N, 1, H, W) or (N, H, W) logits
            targets: (N, 1, H, W) or (N, H, W) binary targets (0 or 1)
        """
        # Ensure inputs are (N, H, W) or (N, 1, H, W) -> flatten appropriately
        if logits.dim() > 2:
            # Check if channel dim is present
            if logits.shape[1] == 1 and logits.dim() == 4:
                logits = logits.squeeze(1)
            if targets.dim() == 4 and targets.shape[1] == 1:
                targets = targets.squeeze(1)

        # Now shapes should be (N, H, W)

        if self.per_image:
            loss = 0.0
            batch_size = logits.size(0)
            for i in range(batch_size):
                # Flatten spatial dims
                logit_flat = logits[i].view(-1)
                target_flat = targets[i].view(-1)
                loss += lovasz_hinge_flat(logit_flat, target_flat)
            return loss / batch_size
        else:
            return lovasz_hinge_flat(logits.view(-1), targets.view(-1))


class BCELovaszLoss(nn.Module):
    """
    Combination of BCEWithLogitsLoss and LovaszHingeLoss.
    Returns the SUM of the two losses.
    """

    def __init__(self):
        super(BCELovaszLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.lovasz = LovaszHingeLoss(per_image=True)

    def forward(self, logits, targets):
        """
        Args:
            logits: (N, 1, H, W)
            targets: (N, 1, H, W)
        """
        # BCEWithLogitsLoss expects float targets
        bce_loss = self.bce(logits, targets.float())
        lovasz_loss = self.lovasz(logits, targets)

        return bce_loss + lovasz_loss
