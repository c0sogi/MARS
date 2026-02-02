import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Jaccard loss w.r.t the sorted error
    See Lovasz-Softmax paper for details
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
    Binary Lovasz hinge loss for segmentation.
    Optimizes the Jaccard index (IoU) directly.
    """

    def __init__(self):
        super(LovaszHingeLoss, self).__init__()

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, 1, H, W) or (B, H, W) logits
            targets: (B, 1, H, W) or (B, H, W) binary masks (0 or 1)
        Returns:
            Scalar loss
        """
        # Squeeze channel dimension if present
        if logits.dim() == 4:
            logits = logits.squeeze(1)
        if targets.dim() == 4:
            targets = targets.squeeze(1)

        batch_size = logits.size(0)
        losses = []

        for i in range(batch_size):
            logit_flat = logits[i].view(-1)
            target_flat = targets[i].view(-1)
            loss = lovasz_hinge_flat(logit_flat, target_flat)
            losses.append(loss)

        return torch.stack(losses).mean()


class CombinedMTLLoss(nn.Module):
    """
    Combined Multi-Task Learning Loss.
    Loss = (BCE + Lovasz) + weight * MSE_Depth
    """

    def __init__(self):
        super(CombinedMTLLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.lovasz = LovaszHingeLoss()
        self.mse = nn.MSELoss()
        self.depth_weight = Config.DEPTH_LOSS_WEIGHT

    def forward(self, outputs, targets):
        """
        Args:
            outputs: Dictionary containing:
                - 'mask': (B, 1, H, W) logits for segmentation
                - 'depth': (B, 1) predicted depth values (if Config.AUX_DEPTH_HEAD is True)
            targets: Dictionary containing:
                - 'mask': (B, 1, H, W) ground truth masks
                - 'depth': (B, 1) ground truth depths
        Returns:
            total_loss: Scalar tensor
            metrics: Dictionary of individual loss components for logging
        """
        pred_mask = outputs["mask"]
        true_mask = targets["mask"]

        # Segmentation Losses
        # BCEWithLogitsLoss expects float targets
        loss_bce = self.bce(pred_mask, true_mask.float())
        loss_lovasz = self.lovasz(pred_mask, true_mask.float())

        seg_loss = loss_bce + loss_lovasz

        # Depth Loss
        loss_depth = torch.tensor(0.0, device=pred_mask.device)

        # Check if depth output exists and compute loss
        if "depth" in outputs and outputs["depth"] is not None:
            pred_depth = outputs["depth"]
            true_depth = targets["depth"]

            # Ensure shapes match for MSE (e.g., (B, 1) vs (B,))
            if pred_depth.shape != true_depth.shape:
                true_depth = true_depth.view_as(pred_depth)

            loss_depth = self.mse(pred_depth, true_depth.float())

        # Total Weighted Loss
        total_loss = seg_loss + (self.depth_weight * loss_depth)

        metrics = {
            "loss_bce": loss_bce.detach(),
            "loss_lovasz": loss_lovasz.detach(),
            "loss_depth": loss_depth.detach(),
            "loss_total": total_loss.detach(),
        }

        return total_loss, metrics
