import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from library.config import Config
from library.utils import encode_boxes, iou2d_nearest


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, prediction_logits, target_labels):
        """
        Args:
            prediction_logits: (N, NumClasses)
            target_labels: (N,) Tensor with 0=background, 1..C=classes, -1=ignore
        """
        # Filter ignore indices
        valid_mask = target_labels != -1
        if not valid_mask.any():
            return torch.tensor(0.0, device=prediction_logits.device)

        preds = prediction_logits[valid_mask]
        targets = target_labels[valid_mask]

        # Sigmoid activation
        p = torch.sigmoid(preds)

        # Create one-hot targets
        # targets contain 0 for bg, 1..C for classes.
        # prediction_logits has C channels.
        # We map target 0 -> all zeros. target k -> index k-1 is 1.
        target_one_hot = torch.zeros_like(preds)

        fg_mask = targets > 0
        if fg_mask.any():
            # Convert 1-based class IDs to 0-based indices
            fg_indices = (targets[fg_mask] - 1).long().unsqueeze(1)
            target_one_hot[fg_mask] = target_one_hot[fg_mask].scatter(
                1, fg_indices, 1.0
            )

        # Focal Loss Calculation
        # p_t = p if y=1 else 1-p
        p_t = p * target_one_hot + (1 - p) * (1 - target_one_hot)

        # alpha_t = alpha if y=1 else 1-alpha
        alpha_t = self.alpha * target_one_hot + (1 - self.alpha) * (1 - target_one_hot)

        # FL = -alpha_t * (1-p_t)^gamma * log(p_t)
        # Using binary_cross_entropy_with_logits for numerical stability in the log part is tricky
        # with the explicit focal term, so we calculate manually or use the BCE formulation.
        # Here we use the explicit form matching the reference logic.

        ce_loss = F.binary_cross_entropy_with_logits(
            preds, target_one_hot, reduction="none"
        )
        loss = alpha_t * ce_loss * ((1 - p_t) ** self.gamma)

        # Normalize by number of positive samples
        num_pos = max(1, fg_mask.sum().item())
        return loss.sum() / num_pos


class SmoothL1Loss(nn.Module):
    def __init__(self, beta=1.0 / 9.0):
        super().__init__()
        self.beta = beta

    def forward(self, prediction, target):
        return F.smooth_l1_loss(prediction, target, reduction="mean", beta=self.beta)


class AnchorMatcher:
    def __init__(
        self, anchors, match_thresholds, unmatch_thresholds, anchor_class_indices
    ):
        """
        Args:
            anchors: (N_a, 7) Tensor
            match_thresholds: (N_a,) Tensor
            unmatch_thresholds: (N_a,) Tensor
            anchor_class_indices: List of sets, defining allowed class IDs for each anchor pattern index.
        """
        self.anchors = anchors
        self.match_thresholds = match_thresholds
        self.unmatch_thresholds = unmatch_thresholds
        self.anchor_class_indices = anchor_class_indices
        self.pattern_len = len(anchor_class_indices)

    def __call__(self, gt_boxes, gt_classes, device):
        num_anchors = self.anchors.shape[0]

        # Targets
        cls_tgt = torch.zeros(num_anchors, dtype=torch.long, device=device)  # 0=bg
        reg_tgt = torch.zeros((num_anchors, 7), dtype=torch.float32, device=device)
        dir_tgt = torch.zeros(num_anchors, dtype=torch.long, device=device)

        if len(gt_boxes) == 0:
            return cls_tgt, reg_tgt, dir_tgt

        # 1. Calculate IoU (N_a, N_gt)
        ious = iou2d_nearest(self.anchors, gt_boxes)

        # 2. Max IoU per anchor
        max_iou_per_anchor, max_iou_idx = ious.max(dim=1)  # (N_a,)

        # 3. Assign Negatives (IoU < unmatch_thresh)
        cls_tgt[max_iou_per_anchor < self.unmatch_thresholds] = 0

        # 4. Assign Ignores (unmatch <= IoU < match)
        cls_tgt[
            (max_iou_per_anchor >= self.unmatch_thresholds)
            & (max_iou_per_anchor < self.match_thresholds)
        ] = -1

        # 5. Potential Positives (IoU >= match)
        pos_mask = max_iou_per_anchor >= self.match_thresholds

        # 6. Force match for max IoU per GT
        max_iou_per_gt, anchor_idx_per_gt = ious.max(dim=0)
        pos_mask[anchor_idx_per_gt] = True

        # 7. Check Class Compatibility
        # Get assigned GT index for potential positives
        assigned_gt_idx = max_iou_idx[pos_mask]
        assigned_classes = gt_classes[assigned_gt_idx]

        # Determine anchor type index (modulo pattern length)
        # We need the indices of the anchors that are currently true in pos_mask
        pos_indices = torch.nonzero(pos_mask, as_tuple=True)[0]
        anchor_types = (pos_indices % self.pattern_len).cpu().numpy()

        valid_pos_subset_indices = []
        gt_classes_cpu = assigned_classes.cpu().numpy()

        for i, (a_type, gt_cls) in enumerate(zip(anchor_types, gt_classes_cpu)):
            if gt_cls in self.anchor_class_indices[a_type]:
                valid_pos_subset_indices.append(i)
            else:
                # Class mismatch: Treat as ignore
                abs_idx = pos_indices[i]
                cls_tgt[abs_idx] = -1

        if not valid_pos_subset_indices:
            return cls_tgt, reg_tgt, dir_tgt

        # Filter to keep only valid positives
        valid_subset = torch.tensor(valid_pos_subset_indices, device=device)
        final_pos_indices = pos_indices[valid_subset]

        # 8. Assign Final Targets
        matched_gt_idx = max_iou_idx[final_pos_indices]

        # Classification
        cls_tgt[final_pos_indices] = gt_classes[matched_gt_idx].long()

        # Regression
        matched_gt_boxes = gt_boxes[matched_gt_idx]
        matched_anchors = self.anchors[final_pos_indices]
        reg_tgt[final_pos_indices] = encode_boxes(matched_gt_boxes, matched_anchors)

        # Direction
        # Placeholder logic: 1 if yaw > 0, else 0
        dir_tgt[final_pos_indices] = (matched_gt_boxes[:, 6] > 0).long()

        return cls_tgt, reg_tgt, dir_tgt


class PointPillarsLoss(nn.Module):
    def __init__(self, anchor_generator):
        super().__init__()
        self.anchor_generator = anchor_generator
        # Register anchors as buffer to save with model but not update with gradients
        self.register_buffer("anchors", anchor_generator.get_anchors())

        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        self.smooth_l1_loss = SmoothL1Loss()

        # Initialize configuration patterns
        self._init_config_patterns()

        self.matcher = None  # Initialized lazily with correct device

    def _init_config_patterns(self):
        """Prepares threshold and class mapping patterns based on Config."""
        configs = Config.ANCHOR_GENERATOR_CONFIG

        anchor_indices = []
        matched_thresholds = []
        unmatched_thresholds = []

        for cfg in configs:
            # Allowed classes for this anchor group
            allowed_classes = set([Config.CLASS_TO_ID[n] for n in cfg["class_names"]])

            m_thresh = cfg["matched_threshold"]
            u_thresh = cfg["unmatched_threshold"]

            # Total anchors in this config group (sizes * rotations)
            num_anchors_in_group = len(cfg["anchor_sizes"]) * len(
                cfg["anchor_rotations"]
            )

            for _ in range(num_anchors_in_group):
                anchor_indices.append(allowed_classes)
                matched_thresholds.append(m_thresh)
                unmatched_thresholds.append(u_thresh)

        self.anchor_class_indices = anchor_indices
        self.matched_thresh_pattern = torch.tensor(
            matched_thresholds, dtype=torch.float32
        )
        self.unmatched_thresh_pattern = torch.tensor(
            unmatched_thresholds, dtype=torch.float32
        )

    def _ensure_matcher(self, device):
        """Lazily initializes the matcher and expands thresholds to the full grid."""
        if self.matcher is None or self.matcher.match_thresholds.device != device:
            num_anchors_total = self.anchors.shape[0]
            pattern_len = len(self.matched_thresh_pattern)
            num_locs = num_anchors_total // pattern_len

            # Expand patterns to full grid
            matched_thresh = self.matched_thresh_pattern.to(device).repeat(num_locs)
            unmatched_thresh = self.unmatched_thresh_pattern.to(device).repeat(num_locs)

            self.matcher = AnchorMatcher(
                self.anchors.to(device),
                matched_thresh,
                unmatched_thresh,
                self.anchor_class_indices,
            )

    def forward(self, cls_preds, box_preds, dir_preds, gt_boxes_list, gt_classes_list):
        """
        Args:
            cls_preds: (B, N_a, NumClasses)
            box_preds: (B, N_a, 7)
            dir_preds: (B, N_a, 2)
            gt_boxes_list: List of (M, 7) tensors
            gt_classes_list: List of (M,) tensors
        """
        device = cls_preds.device
        batch_size = cls_preds.shape[0]
        self._ensure_matcher(device)

        total_cls_loss = 0.0
        total_loc_loss = 0.0
        total_dir_loss = 0.0

        for b in range(batch_size):
            gt_boxes = gt_boxes_list[b].to(device)
            gt_classes = gt_classes_list[b].to(device)

            # 1. Assign Targets
            cls_tgt, reg_tgt, dir_tgt = self.matcher(gt_boxes, gt_classes, device)

            # 2. Classification Loss
            # cls_tgt has 0 for bg, 1..C for classes, -1 for ignore
            cls_loss = self.focal_loss(cls_preds[b], cls_tgt)
            total_cls_loss += cls_loss

            # 3. Regression and Direction Loss
            # Only computed on positives
            pos_mask = cls_tgt > 0
            if pos_mask.any():
                # Box Regression
                loc_loss = self.smooth_l1_loss(
                    box_preds[b][pos_mask], reg_tgt[pos_mask]
                )
                total_loc_loss += loc_loss

                # Direction Classification
                dir_loss = F.cross_entropy(dir_preds[b][pos_mask], dir_tgt[pos_mask])
                total_dir_loss += dir_loss

        # Average over batch
        return {
            "cls_loss": total_cls_loss / batch_size,
            "loc_loss": total_loc_loss / batch_size,
            "dir_loss": total_dir_loss / batch_size,
        }
