import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Jaccard loss w.r.t the sorted errors.
    See Alg. 1 in https://arxiv.org/abs/1705.08790
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
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.0

    # Strict FP32 enforcement for stability
    logits = logits.float()
    labels = labels.float()

    signs = 2.0 * labels - 1.0
    errors = 1.0 - logits * signs
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), grad)
    return loss


class LovaszHingeLoss(nn.Module):
    def __init__(self, per_image=True):
        super().__init__()
        self.per_image = per_image

    def forward(self, logits, labels):
        """
        Args:
            logits: [B, 1, H, W] or [B, H, W]
            labels: [B, 1, H, W] or [B, H, W]
        """
        # Squeeze channel dimension if it exists and is 1
        if logits.dim() == 4 and logits.size(1) == 1:
            logits = logits.squeeze(1)
        if labels.dim() == 4 and labels.size(1) == 1:
            labels = labels.squeeze(1)

        if self.per_image:
            loss = 0
            batch_size = logits.size(0)
            for i in range(batch_size):
                # Flatten per image
                l = logits[i].view(-1)
                t = labels[i].view(-1)
                loss += lovasz_hinge_flat(l, t)
            return loss / batch_size
        else:
            # Flatten whole batch
            return lovasz_hinge_flat(logits.view(-1), labels.view(-1))


class CombinedLoss(nn.Module):
    """
    Combines BCE, Lovasz-Hinge, and Auxiliary MSE Loss.
    """

    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.lovasz = LovaszHingeLoss(per_image=True)
        self.mse = nn.MSELoss()

        self.bce_weight = Config.BCE_WEIGHT
        self.lovasz_weight = Config.LOVASZ_WEIGHT
        self.aux_weight = Config.AUX_WEIGHT

    def forward(self, outputs, mask_targets, depth_targets=None):
        """
        Args:
            outputs: Model output. Can be a tensor (logits) or a tuple (logits, aux_pred).
            mask_targets: Ground truth binary masks.
            depth_targets: Ground truth depth values (normalized).
        """
        aux_pred = None
        if isinstance(outputs, (tuple, list)):
            logits = outputs[0]
            if len(outputs) > 1:
                aux_pred = outputs[1]
        else:
            logits = outputs

        # Ensure targets are float for BCE/Lovasz
        mask_targets = mask_targets.float()

        # 1. Binary Cross Entropy (Pixel-wise convergence)
        loss_bce = self.bce(logits, mask_targets)

        # 2. Lovasz Hinge (IoU Optimization)
        loss_lovasz = self.lovasz(logits, mask_targets)

        # Weighted Sum for Segmentation
        total_loss = (self.bce_weight * loss_bce) + (self.lovasz_weight * loss_lovasz)

        # 3. Auxiliary Depth Loss (Feature Alignment)
        if aux_pred is not None and depth_targets is not None:
            # Flatten to ensure shape match (B, 1) vs (B, 1) or (B,)
            aux_pred_flat = aux_pred.view(-1)
            depth_targets_flat = depth_targets.view(-1)

            # Ensure float32
            depth_targets_flat = depth_targets_flat.float()

            loss_aux = self.mse(aux_pred_flat, depth_targets_flat)
            total_loss += self.aux_weight * loss_aux

        return total_loss
