import torch
import torch.nn as nn
import torch.nn.functional as F


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Jaccard loss w.r.t the sorted element of error
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.cumsum(0)
    union = gts + (1 - gt_sorted).cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = labels != ignore
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss on flattened inputs
    logits: [P] Variable, logits at each pixel (between -\infty and +\infty)
    labels: [P] Tensor, binary ground truth masks (0 or 1)
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
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        loss = 0
        batch_size = logits.size(0)
        for i in range(batch_size):
            logit_flat, label_flat = flatten_binary_scores(logits[i], labels[i], ignore)
            loss += lovasz_hinge_flat(logit_flat, label_flat)
        return loss / batch_size
    else:
        logit_flat, label_flat = flatten_binary_scores(logits, labels, ignore)
        return lovasz_hinge_flat(logit_flat, label_flat)


class LovaszHingeLoss(nn.Module):
    def __init__(self, per_image=True, ignore=None):
        super().__init__()
        self.per_image = per_image
        self.ignore = ignore

    def forward(self, logits, labels):
        """
        logits: (N, 1, H, W) or (N, H, W)
        labels: (N, 1, H, W) or (N, H, W)
        """
        # Squeeze channel dimension if present
        if logits.dim() == 4 and logits.shape[1] == 1:
            logits = logits.squeeze(1)
        if labels.dim() == 4 and labels.shape[1] == 1:
            labels = labels.squeeze(1)

        return lovasz_hinge(
            logits, labels, per_image=self.per_image, ignore=self.ignore
        )


class CombinedLoss(nn.Module):
    def __init__(self, bce_weight=1.0, lovasz_weight=1.0, per_image=True, ignore=None):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.lovasz_loss = LovaszHingeLoss(per_image=per_image, ignore=ignore)
        self.bce_weight = bce_weight
        self.lovasz_weight = lovasz_weight

    def forward(self, logits, labels):
        # BCEWithLogitsLoss handles arbitrary shapes as long as they match
        # Ensure labels are float for BCE
        loss1 = self.bce_loss(logits, labels.float())
        loss2 = self.lovasz_loss(logits, labels)

        return self.bce_weight * loss1 + self.lovasz_weight * loss2
