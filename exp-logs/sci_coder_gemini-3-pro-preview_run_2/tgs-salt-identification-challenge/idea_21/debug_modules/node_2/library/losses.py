import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from library.config import Config

# =============================================================================
# Lovasz-Hinge Loss Implementation
# References: https://github.com/bermanmaxim/LovaszSoftmax
# =============================================================================


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
    Wrapper for Lovasz Hinge Loss.
    """

    def __init__(self, per_image=True, ignore=None):
        super().__init__()
        self.per_image = per_image
        self.ignore = ignore

    def forward(self, logits, labels):
        """
        Args:
            logits: (N, 1, H, W) or (N, H, W) logits
            labels: (N, 1, H, W) or (N, H, W) binary labels (0 or 1)
        """
        if self.per_image:
            # Compute loss per image and average
            batch_size = logits.size(0)
            loss = 0
            for i in range(batch_size):
                # Flatten single image
                l_flat, t_flat = flatten_binary_scores(
                    logits[i], labels[i], self.ignore
                )
                loss += lovasz_hinge_flat(l_flat, t_flat)
            return loss / batch_size
        else:
            # Flatten entire batch
            l_flat, t_flat = flatten_binary_scores(logits, labels, self.ignore)
            return lovasz_hinge_flat(l_flat, t_flat)


# =============================================================================
# Composite Losses
# =============================================================================


class SegmentationLoss(nn.Module):
    """
    Standard segmentation loss combining BCEWithLogitsLoss and LovaszHingeLoss.
    Used for the Teacher model and the supervised component of the Student.
    """

    def __init__(self, bce_weight=1.0, lovasz_weight=1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.lovasz = LovaszHingeLoss(per_image=True)
        self.bce_weight = bce_weight
        self.lovasz_weight = lovasz_weight

    def forward(self, logits, targets):
        # Ensure targets are float for BCE
        targets = targets.float()

        loss_bce = self.bce(logits, targets)
        loss_lovasz = self.lovasz(logits, targets)

        return (self.bce_weight * loss_bce) + (self.lovasz_weight * loss_lovasz)


class DistillationLoss(nn.Module):
    """
    Composite loss for the Multi-Task Student model.
    Combines:
    1. Supervised Segmentation Loss (vs Ground Truth)
    2. Knowledge Distillation Loss (vs Teacher Soft Predictions)
    3. Auxiliary Depth Regression Loss (vs Ground Truth Depth)
    """

    def __init__(
        self, lambda_distill=Config.LAMBDA_DISTILL, lambda_depth=Config.LAMBDA_DEPTH
    ):
        super().__init__()
        self.lambda_distill = lambda_distill
        self.lambda_depth = lambda_depth

        # 1. Supervised Segmentation
        self.seg_loss = SegmentationLoss()

        # 2. Distillation (BCE on soft targets)
        self.distill_loss = nn.BCEWithLogitsLoss()

        # 3. Depth Regression
        self.depth_loss = nn.MSELoss()

    def forward(self, student_logits, student_depth, teacher_logits, masks, depths):
        """
        Args:
            student_logits: (N, 1, H, W) Logits from student segmentation head.
            student_depth: (N, 1) or (N,) Predicted depth from student aux head.
            teacher_logits: (N, 1, H, W) Logits from teacher segmentation head.
            masks: (N, 1, H, W) Ground truth binary masks.
            depths: (N,) Ground truth depths.
        """
        # 1. Supervised Segmentation Loss
        l_seg = self.seg_loss(student_logits, masks)

        # 2. Distillation Loss
        # We treat teacher's sigmoid probabilities as soft targets.
        # BCEWithLogitsLoss takes logits as input and targets as probabilities.
        # We detach teacher logits to ensure no gradients flow back to teacher.
        teacher_probs = torch.sigmoid(teacher_logits.detach())
        l_distill = self.distill_loss(student_logits, teacher_probs)

        # 3. Depth Loss
        # Ensure shapes align. Student might output (N, 1), depths is (N,).
        if student_depth.dim() > 1 and student_depth.shape[1] == 1:
            student_depth = student_depth.squeeze(1)

        # Normalize depths or use raw?
        # Usually depth is standardized before training, but here we assume
        # the model outputs raw/scaled depth matching the target 'depths' tensor.
        # We cast depths to float.
        l_depth = self.depth_loss(student_depth, depths.float())

        # Composite
        total_loss = (
            l_seg + (self.lambda_distill * l_distill) + (self.lambda_depth * l_depth)
        )

        return total_loss, l_seg, l_distill, l_depth
