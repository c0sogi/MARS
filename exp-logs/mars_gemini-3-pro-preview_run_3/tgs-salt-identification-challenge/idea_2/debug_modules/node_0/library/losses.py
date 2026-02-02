import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEDiceLoss(nn.Module):
    """
    Combined Binary Cross Entropy and Dice Loss.
    Used for the initial warm-up stage of training.
    """

    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1.0):
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): Raw model predictions (before sigmoid). Shape (B, 1, H, W) or (B, H, W).
            targets (torch.Tensor): Ground truth binary masks. Shape (B, 1, H, W) or (B, H, W).
        """
        # Ensure inputs are flattened and float
        if logits.dim() > 2:
            logits = logits.view(-1)
        if targets.dim() > 2:
            targets = targets.view(-1)

        targets = targets.float()

        # BCE Loss (numerically stable with logits)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets)

        # Dice Loss
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum()
        union = probs.sum() + targets.sum()

        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice_score

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


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


class LovaszHingeLoss(nn.Module):
    """
    Binary Lovasz hinge loss.
    Optimizes the Jaccard index (IoU) directly. Used for fine-tuning.
    """

    def __init__(self, per_image=True):
        super(LovaszHingeLoss, self).__init__()
        self.per_image = per_image

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): Raw model predictions (before sigmoid). Shape (B, 1, H, W) or (B, H, W).
            targets (torch.Tensor): Ground truth binary masks. Shape (B, 1, H, W) or (B, H, W).
        """
        # Squeeze channel dimension if present: (B, 1, H, W) -> (B, H, W)
        if logits.dim() == 4:
            logits = logits.squeeze(1)
        if targets.dim() == 4:
            targets = targets.squeeze(1)

        if self.per_image:
            loss = 0.0
            batch_size = logits.size(0)
            for i in range(batch_size):
                loss += self.lovasz_hinge_flat(logits[i].view(-1), targets[i].view(-1))
            return loss / batch_size
        else:
            return self.lovasz_hinge_flat(logits.view(-1), targets.view(-1))

    def lovasz_hinge_flat(self, logits, labels):
        """
        Binary Lovasz hinge loss on flattened inputs.
        """
        if len(labels) == 0:
            # Only void pixels, the gradients should be 0
            return logits.sum() * 0.0

        signs = 2.0 * labels.float() - 1.0
        errors = 1.0 - logits * signs
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)

        gt_sorted = labels[perm]
        grad = lovasz_grad(gt_sorted)

        # ELU/ReLU is implicit in the hinge definition, but standard implementation
        # applies ReLU to the errors before dot product with the gradient.
        loss = torch.dot(F.relu(errors_sorted), grad)

        return loss
