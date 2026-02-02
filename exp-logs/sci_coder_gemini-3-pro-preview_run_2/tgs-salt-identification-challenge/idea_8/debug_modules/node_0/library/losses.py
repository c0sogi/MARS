import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


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
      logits: [P] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: void class label
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
    Optimizes the Jaccard index (IoU) directly.
    """

    def __init__(self, per_image=True, ignore=None):
        """
        Args:
            per_image (bool): If True, compute loss per image and average.
                              If False, compute over the flattened batch.
            ignore (int, optional): Label value to ignore.
        """
        super(LovaszHingeLoss, self).__init__()
        self.per_image = per_image
        self.ignore = ignore

    def forward(self, logits, targets):
        """
        Args:
            logits: (N, 1, H, W) or (N, H, W) logits from the model.
            targets: (N, 1, H, W) or (N, H, W) binary masks (0 or 1).
        """
        # Squeeze channel dimension if present
        if logits.dim() > 3 and logits.shape[1] == 1:
            logits = logits.squeeze(1)
        if targets.dim() > 3 and targets.shape[1] == 1:
            targets = targets.squeeze(1)

        if self.per_image:
            loss = 0
            batch_size = logits.size(0)
            for i in range(batch_size):
                # Flatten single image
                l_flat, t_flat = flatten_binary_scores(
                    logits[i], targets[i], ignore=self.ignore
                )
                loss += lovasz_hinge_flat(l_flat, t_flat)
            return loss / batch_size
        else:
            # Flatten entire batch
            l_flat, t_flat = flatten_binary_scores(logits, targets, ignore=self.ignore)
            return lovasz_hinge_flat(l_flat, t_flat)


class MixedLoss(nn.Module):
    """
    Composite loss function combining Binary Cross Entropy (BCE) and Lovasz Hinge Loss.
    BCE provides smooth gradients for convergence, while Lovasz optimizes the IoU metric directly.
    """

    def __init__(self, bce_weight=1.0, lovasz_weight=1.0, per_image=True):
        """
        Args:
            bce_weight (float): Weight for the BCE component.
            lovasz_weight (float): Weight for the Lovasz component.
            per_image (bool): Whether to calculate Lovasz loss per image.
        """
        super(MixedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.lovasz_weight = lovasz_weight

        # BCEWithLogitsLoss combines Sigmoid and BCE for numerical stability
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.lovasz_loss = LovaszHingeLoss(per_image=per_image)

    def forward(self, logits, targets):
        """
        Args:
            logits: (N, 1, H, W) raw scores (before sigmoid).
            targets: (N, 1, H, W) binary ground truth masks.
        """
        # BCE expects float targets
        targets_float = targets.float()

        loss = 0.0

        if self.bce_weight > 0:
            bce = self.bce_loss(logits, targets_float)
            loss += self.bce_weight * bce

        if self.lovasz_weight > 0:
            # Lovasz expects logits, handles internal flattening
            lovasz = self.lovasz_loss(logits, targets_float)
            loss += self.lovasz_weight * lovasz

        return loss
