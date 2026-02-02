import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


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
    errors = 1.0 - logits * signs
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), grad)
    return loss


class LovaszHingeLoss(nn.Module):
    """
    Lovasz Hinge Loss for Binary Segmentation.
    Optimizes the Jaccard index (IoU) directly.
    """

    def __init__(self):
        super(LovaszHingeLoss, self).__init__()

    def forward(self, logits, labels):
        """
        Args:
            logits: (N, 1, H, W) or (N, H, W) logits
            labels: (N, 1, H, W) or (N, H, W) binary labels (0 or 1)
        """
        # Per-image aggregation (Cite solution_lesson_node_00044)
        if logits.dim() == 4:
            logits = logits.squeeze(1)
        if labels.dim() == 4:
            labels = labels.squeeze(1)

        loss = 0.0
        batch_size = logits.size(0)

        for i in range(batch_size):
            loss += lovasz_hinge_flat(logits[i].view(-1), labels[i].view(-1))

        return loss / batch_size


class SegmentationLoss(nn.Module):
    """
    Segmentation Loss: BCEWithLogitsLoss + LovaszHingeLoss
    """

    def __init__(self):
        super(SegmentationLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.lovasz = LovaszHingeLoss()

    def forward(self, pred_mask, true_mask):
        # BCE
        loss_bce = self.bce(pred_mask, true_mask)

        # Lovasz
        loss_lovasz = self.lovasz(pred_mask, true_mask)

        # Total Loss
        total_loss = loss_bce + loss_lovasz

        metrics = {
            "loss_bce": loss_bce.item(),
            "loss_lovasz": loss_lovasz.item(),
            "loss_total": total_loss.item(),
        }

        return total_loss, metrics
