import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from library import config

# =============================================================================
# Lovasz-Hinge Loss Implementation
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


def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
        logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
        labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
        per_image: compute the loss per image instead of per batch
        ignore: void class id
    """
    if per_image:
        loss = mean(
            lovasz_hinge_flat(
                *flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore)
            )
            for log, lab in zip(logits, labels)
        )
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss on flattened inputs
        logits: [P] Variable, logits at each pixel
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


def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = filter(lambda x: not (torch.isnan(x) or torch.isinf(x)), l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == "raise":
            raise ValueError("Empty iterator")
        return empty
    for x in l:
        n += 1
        acc += x
    return acc / n


# =============================================================================
# Composite Loss Classes
# =============================================================================


class SegmentationLoss(nn.Module):
    """
    Standard segmentation loss combining BCEWithLogitsLoss and LovaszHingeLoss.
    Used for the Teacher model and the segmentation component of the Student.
    """

    def __init__(self, bce_weight=1.0, lovasz_weight=1.0):
        super(SegmentationLoss, self).__init__()
        self.bce_weight = bce_weight
        self.lovasz_weight = lovasz_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, 1, H, W)
            targets: (B, 1, H, W)
        """
        # Squeeze channel dim for lovasz which expects (B, H, W) usually,
        # or handle inside. Our lovasz_hinge handles (B, H, W).
        # BCE expects (B, 1, H, W) or (B, H, W) if matching.

        # Ensure targets are float for BCE
        targets = targets.float()

        # BCE Loss
        bce_loss = self.bce(logits, targets)

        # Lovasz Loss (expects logits squeezed if channel=1)
        # logits: (B, 1, H, W) -> (B, H, W)
        # targets: (B, 1, H, W) -> (B, H, W)
        logits_sq = logits.squeeze(1)
        targets_sq = targets.squeeze(1)

        lov_loss = lovasz_hinge(logits_sq, targets_sq, per_image=True)

        return self.bce_weight * bce_loss + self.lovasz_weight * lov_loss


class StudentLoss(nn.Module):
    """
    Composite loss for the Multi-Task Student model.
    Includes:
    1. Segmentation Loss (Ground Truth)
    2. Distillation Loss (Teacher Soft Targets)
    3. Depth Regression Loss (Auxiliary Head)
    """

    def __init__(self):
        super(StudentLoss, self).__init__()
        self.seg_loss_fn = SegmentationLoss()
        self.distill_loss_fn = nn.BCEWithLogitsLoss()
        self.depth_loss_fn = nn.MSELoss()

        # Weights from config
        self.w_seg = config.LAMBDA_SEG
        self.w_distill = config.LAMBDA_DISTILL
        self.w_depth = config.LAMBDA_DEPTH

    def forward(
        self, student_logits, teacher_logits, student_depth, targets, true_depth
    ):
        """
        Args:
            student_logits: (B, 1, H, W) - Raw output from student segmentation head
            teacher_logits: (B, 1, H, W) - Raw output from teacher segmentation head
            student_depth: (B, 1) - Raw output from student depth head
            targets: (B, 1, H, W) - Ground truth masks
            true_depth: (B, 1) - Ground truth depth values
        """
        # 1. Segmentation Loss: Student vs Ground Truth
        l_seg = self.seg_loss_fn(student_logits, targets)

        # 2. Distillation Loss: Student vs Teacher
        # We use BCEWithLogitsLoss. The target is sigmoid(teacher_logits).
        # This penalizes the student for diverging from the teacher's probability distribution.
        with torch.no_grad():
            teacher_probs = torch.sigmoid(teacher_logits)

        l_distill = self.distill_loss_fn(student_logits, teacher_probs)

        # 3. Depth Loss: Student Depth vs True Depth
        # Ensure shapes match
        if student_depth.shape != true_depth.shape:
            true_depth = true_depth.view_as(student_depth)

        l_depth = self.depth_loss_fn(student_depth, true_depth.float())

        # Combine
        total_loss = (
            (self.w_seg * l_seg)
            + (self.w_distill * l_distill)
            + (self.w_depth * l_depth)
        )

        return total_loss, {
            "loss_seg": l_seg.item(),
            "loss_distill": l_distill.item(),
            "loss_depth": l_depth.item(),
        }
