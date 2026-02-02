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
      ignore: label to ignore
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
    This is a surrogate for the Jaccard (IoU) loss.
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

    def forward(self, logits, labels):
        """
        Args:
            logits: (B, C, H, W) or (B, H, W) tensor. C must be 1 for binary.
            labels: (B, H, W) or (B, 1, H, W) tensor.
        """
        # Squeeze channel dim if present
        if logits.dim() == 4:
            logits = logits.squeeze(1)
        if labels.dim() == 4:
            labels = labels.squeeze(1)

        if self.per_image:
            batch_size = logits.size(0)
            loss = 0
            for i in range(batch_size):
                l_flat, t_flat = flatten_binary_scores(
                    logits[i], labels[i], self.ignore
                )
                loss += lovasz_hinge_flat(l_flat, t_flat)
            return loss / batch_size
        else:
            l_flat, t_flat = flatten_binary_scores(logits, labels, self.ignore)
            return lovasz_hinge_flat(l_flat, t_flat)


class CombinedLoss(nn.Module):
    """
    Compound loss function: Sum of Lovasz-Hinge Loss and Binary Cross Entropy.
    The summation maintains gradient magnitude for effective optimization.
    """

    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.lovasz = LovaszHingeLoss(per_image=True)

    def forward(self, logits, labels):
        """
        Args:
            logits: (B, 1, H, W) Raw logits from the network.
            labels: (B, 1, H, W) Binary ground truth masks.
        """
        # BCEWithLogitsLoss expects float targets
        bce_loss = self.bce(logits, labels.float())

        # Lovasz expects binary targets (0/1), usually handled internally,
        # but we pass the raw tensors. The LovaszHingeLoss class handles flattening.
        lovasz_loss = self.lovasz(logits, labels)

        return bce_loss + lovasz_loss
