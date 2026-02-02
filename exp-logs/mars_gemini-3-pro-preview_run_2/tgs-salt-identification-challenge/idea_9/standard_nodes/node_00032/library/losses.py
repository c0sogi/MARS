import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# Lovasz-Softmax / Hinge Loss Helpers
# Adapted from: https://github.com/bermanmaxim/LovaszSoftmax
# -----------------------------------------------------------------------------


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension of jaccard index.
    gt_sorted: 1 if label is 1, 0 if label is 0
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.cumsum(0)
    union = gts + (1 - gt_sorted).cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


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


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss on flattened tensors
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


def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        loss = 0
        batch_size = logits.size(0)
        for log, lab in zip(logits, labels):
            # Flatten per image
            flat_log, flat_lab = flatten_binary_scores(
                log.unsqueeze(0), lab.unsqueeze(0), ignore
            )
            loss += lovasz_hinge_flat(flat_log, flat_lab)
        return loss / batch_size
    else:
        flat_log, flat_lab = flatten_binary_scores(logits, labels, ignore)
        return lovasz_hinge_flat(flat_log, flat_lab)


# -----------------------------------------------------------------------------
# Loss Modules
# -----------------------------------------------------------------------------


class LovaszHingeLoss(nn.Module):
    """
    Wrapper for Lovasz Hinge Loss.
    """

    def __init__(self, per_image=True, ignore=None):
        super().__init__()
        self.per_image = per_image
        self.ignore = ignore

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, 1, H, W) or (B, H, W) logits from the model.
            targets: (B, 1, H, W) or (B, H, W) binary masks (0 or 1).
        """
        # Squeeze channel dim if present to match (B, H, W) expected by lovasz_hinge
        if logits.dim() == 4 and logits.size(1) == 1:
            logits = logits.squeeze(1)
        if targets.dim() == 4 and targets.size(1) == 1:
            targets = targets.squeeze(1)

        return lovasz_hinge(
            logits, targets, per_image=self.per_image, ignore=self.ignore
        )


class MixedLoss(nn.Module):
    """
    Combines BCEWithLogitsLoss and LovaszHingeLoss.
    Used for Phase 1 (Teacher) and as the segmentation component in Phase 2.
    """

    def __init__(self, bce_weight=1.0, lovasz_weight=1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.lovasz_weight = lovasz_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.lovasz = LovaszHingeLoss()

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, 1, H, W)
            targets: (B, 1, H, W)
        """
        # BCE expects float targets
        targets_float = targets.float()

        loss_bce = self.bce(logits, targets_float)
        loss_lovasz = self.lovasz(logits, targets)

        return self.bce_weight * loss_bce + self.lovasz_weight * loss_lovasz


class DistillationLoss(nn.Module):
    """
    Composite loss for Phase 2 (Student).
    L = seg_weight * L_Seg(Student, GT) + dist_weight * L_MSE(Student_Logits, Teacher_Logits)
    """

    def __init__(
        self, seg_weight=0.5, dist_weight=0.5, bce_weight=1.0, lovasz_weight=1.0
    ):
        super().__init__()
        self.seg_weight = seg_weight
        self.dist_weight = dist_weight
        # The segmentation loss component (Student vs Ground Truth)
        self.seg_loss = MixedLoss(bce_weight=bce_weight, lovasz_weight=lovasz_weight)
        # The distillation loss component (Student vs Teacher)
        self.mse_loss = nn.MSELoss()

    def forward(self, student_logits, teacher_logits, targets):
        """
        Args:
            student_logits: (B, 1, H, W)
            teacher_logits: (B, 1, H, W)
            targets: (B, 1, H, W)
        """
        # 1. Segmentation Loss: Student vs Ground Truth
        l_seg = self.seg_loss(student_logits, targets)

        # 2. Distillation Loss: Student Logits vs Teacher Logits
        # Important: Detach teacher logits to stop gradients flowing back to teacher
        l_mse = self.mse_loss(student_logits, teacher_logits.detach())

        return self.seg_weight * l_seg + self.dist_weight * l_mse
