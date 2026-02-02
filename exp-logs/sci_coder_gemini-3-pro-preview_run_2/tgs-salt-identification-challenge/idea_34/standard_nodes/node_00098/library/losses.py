import torch
import torch.nn as nn
import torch.nn.functional as F


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Jaccard extension w.r.t the sort of the errors
    gt_sorted: binary ground truth sorted by error
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

    # The gradient of the loss is the dot product of the error and the Jaccard gradient
    loss = torch.dot(F.relu(errors_sorted), grad)
    return loss


class LovaszHingeLoss(nn.Module):
    """
    Lovasz Hinge Loss for binary segmentation.
    Optimizes the Jaccard index (IoU) directly.
    """

    def __init__(self, per_image=True, ignore_index=None):
        super(LovaszHingeLoss, self).__init__()
        self.per_image = per_image
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, 1, H, W) or (B, H, W) logits
            targets: (B, 1, H, W) or (B, H, W) binary targets (0 or 1)
        """
        # Squeeze channel dim if present
        if logits.dim() > 3:
            logits = logits.squeeze(1)
        if targets.dim() > 3:
            targets = targets.squeeze(1)

        if self.per_image:
            loss = 0
            batch_size = logits.size(0)
            for i in range(batch_size):
                l = logits[i].view(-1)
                t = targets[i].view(-1)

                if self.ignore_index is not None:
                    valid = t != self.ignore_index
                    l = l[valid]
                    t = t[valid]

                loss += lovasz_hinge_flat(l, t)
            return loss / batch_size
        else:
            l = logits.view(-1)
            t = targets.view(-1)
            if self.ignore_index is not None:
                valid = t != self.ignore_index
                l = l[valid]
                t = t[valid]
            return lovasz_hinge_flat(l, t)
