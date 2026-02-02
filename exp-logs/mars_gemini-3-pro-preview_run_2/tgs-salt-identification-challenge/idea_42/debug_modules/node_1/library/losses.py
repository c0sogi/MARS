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
        super().__init__()

    def forward(self, logits, labels):
        """
        Args:
            logits: (N, C, H, W) or (N, H, W)
            labels: (N, H, W)
        """
        # Flatten inputs
        logits = logits.reshape(-1)
        labels = labels.reshape(-1)
        return lovasz_hinge_flat(logits, labels)


class CompositeLoss(nn.Module):
    """
    Composite Loss function orchestrating the training objectives.

    Modes:
    - 'supervised':
        Mask: BCEWithLogitsLoss + LovaszHingeLoss
        Depth: MSELoss (if auxiliary head is active)
    - 'unlabeled':
        Mask: BCEWithLogitsLoss (against soft targets)
        Depth: Ignored
    """

    def __init__(self, config=Config):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.lovasz = LovaszHingeLoss()
        self.mse = nn.MSELoss()

        self.w_bce = config.LOSS_WEIGHT_BCE
        self.w_lovasz = config.LOSS_WEIGHT_LOVASZ
        self.w_mse = config.LOSS_WEIGHT_MSE

    def forward(self, outputs, targets, mode="supervised"):
        """
        Args:
            outputs: Tensor (mask_logits) OR Dict {'mask': logits, 'depth': val}
            targets: Tensor (mask_gt) OR Dict {'mask': gt, 'depth': val}
            mode: 'supervised' (hard targets) or 'unlabeled' (soft targets)

        Returns:
            loss: scalar tensor
            loss_dict: dictionary of individual loss components for logging
        """
        # 1. Parse Outputs
        if isinstance(outputs, dict):
            mask_logits = outputs["mask"]
            depth_pred = outputs.get("depth", None)
        else:
            mask_logits = outputs
            depth_pred = None

        # 2. Parse Targets
        if isinstance(targets, dict):
            mask_target = targets["mask"]
            depth_target = targets.get("depth", None)
        else:
            mask_target = targets
            depth_target = None

        loss = 0.0
        loss_dict = {}

        # Ensure mask_logits and mask_target shapes align for BCE
        # If targets is (N, H, W), unsqueeze to (N, 1, H, W) if logits is 4D
        if mask_logits.dim() == 4 and mask_target.dim() == 3:
            mask_target = mask_target.unsqueeze(1)

        # 3. Calculate Mask Loss
        if mode == "unlabeled":
            # Soft Targets: Use BCE only (Lovasz is not compatible with soft targets)
            # mask_target is expected to be soft probabilities here
            curr_bce = self.bce(mask_logits, mask_target)
            loss += self.w_bce * curr_bce
            loss_dict["bce_soft"] = curr_bce.item()

        else:  # supervised
            # Hard Targets: Use BCE + Lovasz
            # Ensure targets are float for BCE/Lovasz
            mask_target_float = mask_target.float()

            curr_bce = self.bce(mask_logits, mask_target_float)
            curr_lovasz = self.lovasz(mask_logits, mask_target_float)

            loss += (self.w_bce * curr_bce) + (self.w_lovasz * curr_lovasz)
            loss_dict["bce"] = curr_bce.item()
            loss_dict["lovasz"] = curr_lovasz.item()

        # 4. Calculate Depth Loss (Auxiliary)
        # Only apply if depth prediction and target exist, and we are in supervised mode
        if depth_pred is not None and depth_target is not None and mode == "supervised":
            # Flatten to ensure shape match (N,) vs (N,)
            d_pred = depth_pred.view(-1)
            d_true = depth_target.view(-1)

            curr_mse = self.mse(d_pred, d_true)
            loss += self.w_mse * curr_mse
            loss_dict["mse_depth"] = curr_mse.item()

        return loss, loss_dict
