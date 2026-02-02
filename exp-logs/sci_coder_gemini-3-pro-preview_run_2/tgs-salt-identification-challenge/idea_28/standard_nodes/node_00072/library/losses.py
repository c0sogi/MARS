import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Jaccard loss w.r.t the sorted error
    See Alg. 1 in paper: https://arxiv.org/abs/1705.08790
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


class LovaszHingeLoss(nn.Module):
    """
    Binary Lovasz hinge loss implementation.
    """

    def __init__(self, per_image=True):
        super().__init__()
        self.per_image = per_image

    def forward(self, logits, labels):
        """
        Args:
            logits: [B, H, W] Logits at each pixel (between -\infty and +\infty)
            labels: [B, H, W] Binary ground truth labels (0 or 1)
        """
        if self.per_image:
            loss = 0
            # Iterate over batch dimension to compute loss per image
            for input, target in zip(logits, labels):
                input_flat = input.view(-1)
                target_flat = target.view(-1)
                loss += lovasz_hinge_flat(input_flat, target_flat)
            return loss / logits.size(0)
        else:
            return lovasz_hinge_flat(logits.view(-1), labels.view(-1))


class SegmentationLoss(nn.Module):
    """
    Combined loss function: BCE + Lovasz Hinge.
    Cite solution_lesson_node_00021: Lovasz-Hinge is superior for IoU metrics.
    """

    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.lovasz = LovaszHingeLoss(per_image=True)

    def forward(self, pred_mask, true_mask):
        """
        Args:
            pred_mask: (B, 1, H, W) Logits from the segmentation head.
            true_mask: (B, H, W) or (B, 1, H, W) Ground truth masks.

        Returns:
            total_loss: Sum of BCE and Lovasz.
            metrics: Dictionary containing individual loss components.
        """
        # 1. Prepare Masks
        # Ensure true_mask matches pred_mask shape for BCE (B, 1, H, W)
        if true_mask.dim() == 3:
            true_mask_bce = true_mask.unsqueeze(1)
        else:
            true_mask_bce = true_mask

        # Ensure inputs are squeezed for Lovasz (B, H, W)
        if pred_mask.dim() == 4:
            pred_mask_lovasz = pred_mask.squeeze(1)
        else:
            pred_mask_lovasz = pred_mask

        if true_mask.dim() == 4:
            true_mask_lovasz = true_mask.squeeze(1)
        else:
            true_mask_lovasz = true_mask

        # 2. Compute Segmentation Losses
        # BCEWithLogitsLoss
        loss_bce = self.bce(pred_mask, true_mask_bce.float())

        # Lovasz Hinge Loss (Per Image)
        loss_lovasz = self.lovasz(pred_mask_lovasz, true_mask_lovasz)

        # 3. Combine
        # Summing terms (Cite solution_lesson_node_00036: Summing vs Averaging affects LR)
        total_loss = loss_bce + loss_lovasz

        return total_loss, {
            "loss_bce": loss_bce.item(),
            "loss_lovasz": loss_lovasz.item(),
            "loss_total": total_loss.item(),
        }
