import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class BCEDiceLoss(nn.Module):
    """
    Combination of Binary Cross Entropy and Dice Loss.
    Useful for stable initial training and handling class imbalance.
    """

    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1.0):
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Model output logits (N, C, H, W) or (N, H, W)
            targets: Ground truth masks (N, C, H, W) or (N, H, W), range [0, 1]
        """
        # BCE Loss (Pixel-wise, so flattening is fine)
        loss_bce = self.bce_loss(inputs.view(-1), targets.view(-1))

        # Dice Loss (Sample-wise to handle empty masks correctly)
        inputs_soft = torch.sigmoid(inputs)

        # Flatten spatial dimensions (H, W) but preserve batch dimension (N)
        # inputs: (N, ...) -> (N, -1)
        inputs_flat = inputs_soft.view(inputs_soft.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)

        intersection = (inputs_flat * targets_flat).sum(dim=1)
        union = inputs_flat.sum(dim=1) + targets_flat.sum(dim=1)

        # Calculate Dice per sample
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Average Dice Loss across the batch
        loss_dice = 1 - dice_score.mean()

        return self.bce_weight * loss_bce + self.dice_weight * loss_dice


# ---------------------------------------------------------------------------
# Lovasz-Softmax and Lovasz-Hinge losses
# Reference: https://github.com/bermanmaxim/LovaszSoftmax
# ---------------------------------------------------------------------------


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
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
    errors = 1.0 - logits * Variable(signs)
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), Variable(grad))
    return loss


class LovaszHingeLoss(nn.Module):
    """
    Lovasz-Hinge loss for binary segmentation.
    Optimizes the Jaccard index (IoU) directly.
    """

    def __init__(self, weight=None, ignore_index=None):
        super(LovaszHingeLoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        """
        Args:
            logits: (N, 1, H, W) or (N, H, W) logits
            targets: (N, 1, H, W) or (N, H, W) binary targets {0, 1}
        """
        # Squeeze channel dim if present
        if logits.dim() == 4:
            logits = logits.squeeze(1)
        if targets.dim() == 4:
            targets = targets.squeeze(1)

        logits_flat = logits.view(-1)
        targets_flat = targets.view(-1)

        return lovasz_hinge_flat(logits_flat, targets_flat)


class DeepSupervisionLoss(nn.Module):
    """
    Wrapper to apply a base loss function to multiple output heads (Deep Supervision).
    Automatically resizes ground truth to match the spatial resolution of each output.
    """

    def __init__(self, base_loss, weights=None):
        """
        Args:
            base_loss: The loss module to apply (e.g., BCEDiceLoss).
            weights: List of float weights for each output head.
                     If None, assumes equal weighting or single output.
        """
        super(DeepSupervisionLoss, self).__init__()
        self.base_loss = base_loss
        self.weights = weights

    def forward(self, preds, target):
        """
        Args:
            preds: List/Tuple of tensors [output_head_1, output_head_2, ...]
                   or a single tensor.
            target: Ground truth tensor (N, C, H, W) or (N, H, W).
        """
        # Handle single output case
        if not isinstance(preds, (list, tuple)):
            return self.base_loss(preds, target)

        if self.weights is None:
            # Default to equal weights if not provided, or 1.0 for the first and 0 for others?
            # Usually deep supervision implies we want to train all.
            # Let's default to equal weights summing to 1 for safety,
            # though usually the user should provide specific weights.
            self.weights = [1.0 / len(preds)] * len(preds)

        assert len(preds) == len(
            self.weights
        ), f"Number of predictions ({len(preds)}) must match number of weights ({len(self.weights)})"

        total_loss = 0.0

        for i, pred in enumerate(preds):
            weight = self.weights[i]
            if weight <= 0:
                continue

            # Check dimensions
            # pred: (N, C, H_p, W_p) or (N, H_p, W_p)
            # target: (N, C, H_t, W_t) or (N, H_t, W_t)

            # Determine spatial size of prediction
            if pred.dim() == 4:
                _, _, h_p, w_p = pred.shape
            else:
                _, h_p, w_p = pred.shape

            # Resize target if necessary
            # We use interpolate. Target should be float for interpolation usually,
            # but masks are binary. Nearest neighbor is appropriate for masks.
            if target.dim() == 4:
                _, _, h_t, w_t = target.shape
            else:
                _, h_t, w_t = target.shape

            current_target = target
            if (h_p != h_t) or (w_p != w_t):
                # Ensure target has channel dim for interpolation
                if current_target.dim() == 3:
                    current_target = current_target.unsqueeze(1)

                current_target = F.interpolate(
                    current_target.float(), size=(h_p, w_p), mode="nearest"
                )

                # Remove channel dim if input didn't have it
                if target.dim() == 3:
                    current_target = current_target.squeeze(1)

            # Calculate loss for this head
            loss = self.base_loss(pred, current_target)
            total_loss += weight * loss

        return total_loss
