import torch
import torch.nn as nn
import torch.nn.functional as F


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Jaccard loss w.r.t the sorted error
    See Alg. 1 in https://arxiv.org/abs/1705.08790
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


class LovaszHingeLoss(nn.Module):
    """
    Lovasz Hinge Loss for Binary Segmentation.
    Optimizes the Jaccard index (IoU) directly using the Lovasz extension.
    """

    def __init__(self, per_image=True, ignore=None):
        super(LovaszHingeLoss, self).__init__()
        self.per_image = per_image
        self.ignore = ignore

    def forward(self, logits, targets):
        """
        Args:
            logits: (N, 1, H, W) or (N, H, W) logits from the model.
            targets: (N, 1, H, W) or (N, H, W) binary ground truth masks.
        """
        if self.per_image:
            # Compute loss per image and average
            batch_size = logits.size(0)
            loss = 0
            for i in range(batch_size):
                # Slice and flatten
                logit_flat, target_flat = flatten_binary_scores(
                    logits[i], targets[i], self.ignore
                )
                loss += lovasz_hinge_flat(logit_flat, target_flat)
            return loss / batch_size
        else:
            logit_flat, target_flat = flatten_binary_scores(
                logits, targets, self.ignore
            )
            return lovasz_hinge_flat(logit_flat, target_flat)


class StableBCELoss(nn.Module):
    """
    Stable Binary Cross Entropy Loss using BCEWithLogitsLoss.
    Suitable for binary segmentation and soft-target distillation.
    """

    def __init__(self, reduction="mean"):
        super(StableBCELoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, logits, targets):
        """
        Args:
            logits: (N, 1, H, W) logits.
            targets: (N, 1, H, W) binary or soft targets.
        """
        # Ensure shapes match exactly
        if logits.shape != targets.shape:
            # If targets are (N, H, W) and logits (N, 1, H, W), unsqueeze targets
            if logits.dim() == 4 and targets.dim() == 3:
                targets = targets.unsqueeze(1)
            # If targets are (N, 1, H, W) and logits (N, H, W), unsqueeze logits
            elif logits.dim() == 3 and targets.dim() == 4:
                logits = logits.unsqueeze(1)

        return self.bce(logits, targets.float())


class CombinedLoss(nn.Module):
    """
    Combined Loss: Sum of Lovasz Hinge Loss and Stable BCE Loss.
    Used for stabilizing training by providing both pixel-wise and structural supervision.
    """

    def __init__(self, bce_weight=1.0, lovasz_weight=1.0, per_image=True):
        super(CombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.lovasz_weight = lovasz_weight
        self.bce = StableBCELoss()
        self.lovasz = LovaszHingeLoss(per_image=per_image)

    def forward(self, logits, targets):
        loss = 0.0
        if self.bce_weight > 0:
            loss += self.bce_weight * self.bce(logits, targets)
        if self.lovasz_weight > 0:
            loss += self.lovasz_weight * self.lovasz(logits, targets)
        return loss


class AuxiliaryMSELoss(nn.Module):
    """
    Mean Squared Error Loss for the auxiliary depth regression head.
    """

    def __init__(self):
        super(AuxiliaryMSELoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, preds, targets):
        """
        Args:
            preds: (N, 1) or (N,) predicted depths.
            targets: (N, 1) or (N,) ground truth depths.
        """
        # Ensure inputs are float
        preds = preds.float()
        targets = targets.float()

        # Flatten to ensure shape matching (N,)
        return self.mse(preds.view(-1), targets.view(-1))
