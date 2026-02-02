import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from library.config import (
    ANCHOR_CONFIGS,
    LOSS_WEIGHTS,
    POINT_CLOUD_RANGE,
    VOXEL_SIZE,
    GRID_SIZE,
)
from library.utils import box_encode, box_iou_3d_pair


class PointPillarsLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cls_weight = LOSS_WEIGHTS.get("cls_weight", 1.0)
        self.box_weight = LOSS_WEIGHTS.get("box_weight", 2.0)
        self.dir_weight = LOSS_WEIGHTS.get("dir_weight", 0.2)

        # Focal Loss Settings
        self.alpha = 0.25
        self.gamma = 2.0

        # Anchor Info
        self.num_classes = len(ANCHOR_CONFIGS)
        self.rots_per_class = 2  # Assuming 2 rotations per class as per config
        self.num_types = self.num_classes * self.rots_per_class

    def forward(self, cls_preds, box_preds, dir_preds, anchors, batch_dict):
        """
        Args:
            cls_preds: (B, H*W*Num_Anchors, 1) - Raw logits
            box_preds: (B, H*W*Num_Anchors, 7)
            dir_preds: (B, H*W*Num_Anchors, 2)
            anchors: (H*W*Num_Anchors, 7)
            batch_dict: dict containing 'gt_boxes', 'gt_labels'
        Returns:
            loss_dict: dict containing calculated losses
        """
        gt_boxes_list = batch_dict["gt_boxes"]
        gt_labels_list = batch_dict["gt_labels"]
        batch_size = len(gt_boxes_list)

        # Ensure anchors are on the correct device
        anchors = anchors.to(cls_preds.device)

        total_cls_loss = 0.0
        total_box_loss = 0.0
        total_dir_loss = 0.0

        # Process each sample in the batch
        for b in range(batch_size):
            gt_boxes = gt_boxes_list[b].to(cls_preds.device)
            gt_labels = gt_labels_list[b].to(cls_preds.device)

            b_cls_preds = cls_preds[b]
            b_box_preds = box_preds[b]
            b_dir_preds = dir_preds[b]

            # Generate Targets
            cls_targets, reg_targets, dir_targets = self.assign_targets(
                anchors, gt_boxes, gt_labels
            )

            # 1. Classification Loss (Focal Loss)
            # cls_targets: 0 (neg), 1 (pos), -1 (ignore)

            # Flatten
            probs = torch.sigmoid(b_cls_preds).view(-1)
            targets = cls_targets.view(-1)

            # Filter ignore (-1)
            valid_mask = targets != -1
            if valid_mask.sum() > 0:
                valid_probs = probs[valid_mask]
                valid_targets = targets[valid_mask]

                # Focal Loss
                # pt = p if y=1 else 1-p
                pt = torch.where(valid_targets == 1, valid_probs, 1 - valid_probs)
                alpha_t = torch.where(valid_targets == 1, self.alpha, 1 - self.alpha)
                focal_weight = alpha_t * (1 - pt).pow(self.gamma)

                bce_loss = -(torch.log(pt + 1e-6))
                cls_loss = focal_weight * bce_loss

                # Normalize by number of positive anchors
                num_pos = (targets == 1).sum()
                if num_pos > 0:
                    total_cls_loss += cls_loss.sum() / num_pos
                else:
                    # If no positives, average over all valid anchors to prevent explosion
                    total_cls_loss += cls_loss.mean()

            # 2. Regression Loss
            # Only for positive anchors
            pos_mask = targets == 1
            if pos_mask.sum() > 0:
                pos_reg_preds = b_box_preds[pos_mask]
                pos_reg_targets = reg_targets[pos_mask]

                reg_loss = F.smooth_l1_loss(
                    pos_reg_preds, pos_reg_targets, reduction="mean"
                )
                total_box_loss += reg_loss

                # 3. Direction Loss
                pos_dir_preds = b_dir_preds[pos_mask]
                pos_dir_targets = dir_targets[pos_mask]

                dir_loss = F.cross_entropy(
                    pos_dir_preds, pos_dir_targets, reduction="mean"
                )
                total_dir_loss += dir_loss

        # Average over batch
        loss_dict = {
            "cls_loss": (total_cls_loss / batch_size) * self.cls_weight,
            "box_loss": (total_box_loss / batch_size) * self.box_weight,
            "dir_loss": (total_dir_loss / batch_size) * self.dir_weight,
        }
        loss_dict["loss"] = (
            loss_dict["cls_loss"] + loss_dict["box_loss"] + loss_dict["dir_loss"]
        )

        return loss_dict

    def assign_targets(self, anchors, gt_boxes, gt_labels):
        """
        Assigns anchors to GT boxes based on IoU and class matching.

        Args:
            anchors: (N_anchors, 7)
            gt_boxes: (N_gt, 7)
            gt_labels: (N_gt,)

        Returns:
            cls_targets: (N_anchors,) {-1, 0, 1}
            reg_targets: (N_anchors, 7)
            dir_targets: (N_anchors,) {0, 1}
        """
        num_anchors = anchors.shape[0]
        device = anchors.device

        # Initialize targets
        # 0: Background, 1: Foreground, -1: Ignore
        cls_targets = torch.zeros(num_anchors, dtype=torch.float32, device=device)
        reg_targets = torch.zeros((num_anchors, 7), dtype=torch.float32, device=device)
        dir_targets = torch.zeros(num_anchors, dtype=torch.long, device=device)

        # Keep track of max IoU per anchor to handle overlapping GTs correctly
        anchor_max_iou = torch.zeros(num_anchors, dtype=torch.float32, device=device)

        if len(gt_boxes) == 0:
            return cls_targets, reg_targets, dir_targets

        # Determine Class of each Anchor
        # Anchor index i -> Type t = i % num_types -> Class c = t // 2
        anchor_indices = torch.arange(num_anchors, device=device)
        anchor_type_indices = anchor_indices % self.num_types
        anchor_class_indices = anchor_type_indices // 2

        # Optimization: Filter anchors by distance
        # Calculate centers
        anchor_centers = anchors[:, :2]
        gt_centers = gt_boxes[:, :2]

        # Distance matrix (N_anchors, N_gt)
        # 400k x 50 is manageable on GPU
        dists = torch.cdist(anchor_centers, gt_centers)

        # Threshold for consideration (meters)
        # Only check IoU for anchors close to GT
        dist_thresh = 5.0

        # Convert to CPU/Numpy for IoU calculation (utils function is CPU based)
        anchors_np = anchors.detach().cpu().numpy()
        gt_boxes_np = gt_boxes.detach().cpu().numpy()

        # Iterate over each GT object
        for i in range(len(gt_boxes)):
            gt = gt_boxes_np[i]
            gt_cls = int(gt_labels[i].item()) - 1  # 0-indexed class ID

            # Find candidate anchors:
            # 1. Within distance threshold
            # 2. Matching class ID
            dist_mask = dists[:, i] < dist_thresh
            cls_mask = anchor_class_indices == gt_cls
            candidate_mask = dist_mask & cls_mask

            candidate_idxs = torch.nonzero(candidate_mask, as_tuple=True)[0]

            if len(candidate_idxs) == 0:
                continue

            # Calculate IoU for candidates
            cand_anchors_np = anchors_np[candidate_idxs.cpu().numpy()]

            # List comprehension for IoU (utils function is scalar)
            ious = [box_iou_3d_pair(ca, gt) for ca in cand_anchors_np]
            ious = torch.tensor(ious, device=device, dtype=torch.float32)

            if len(ious) == 0:
                continue

            # Get config for this class
            cfg = ANCHOR_CONFIGS[gt_cls]
            pos_thresh = cfg["matched_threshold"]
            neg_thresh = cfg["unmatched_threshold"]

            # Identify Positive and Ignore
            is_pos = ious >= pos_thresh
            is_ignore = (ious >= neg_thresh) & (ious < pos_thresh)

            # Global indices
            pos_indices = candidate_idxs[is_pos]
            ignore_indices = candidate_idxs[is_ignore]

            # Update Logic: Only update if this GT has higher IoU than previous assignment
            # This handles cases where an anchor overlaps multiple GTs
            current_max_ious = anchor_max_iou[candidate_idxs]
            update_mask = ious > current_max_ious

            # Update Max IoU record
            anchor_max_iou[candidate_idxs[update_mask]] = ious[update_mask]

            # Apply updates
            valid_pos = pos_indices[update_mask[is_pos]]
            valid_ignore = ignore_indices[update_mask[is_ignore]]

            # Set Positive Targets
            if len(valid_pos) > 0:
                cls_targets[valid_pos] = 1

                # Regression Targets
                gt_repeated = gt_boxes[i].unsqueeze(0).repeat(len(valid_pos), 1)
                encoded = box_encode(gt_repeated, anchors[valid_pos])
                reg_targets[valid_pos] = encoded

                # Direction Targets
                # 1 if orientation difference is positive (sin > 0)
                anchor_yaws = anchors[valid_pos, 6]
                gt_yaw = gt_boxes[i, 6]
                dir_mask = torch.sin(gt_yaw - anchor_yaws) > 0
                dir_targets[valid_pos] = dir_mask.long()

            # Set Ignore Targets
            if len(valid_ignore) > 0:
                # Set to -1 only if not already positive (handled by update logic implicitly)
                # We assume if it was positive for another GT, max_iou would be higher,
                # so update_mask would be false here if this iou is lower.
                cls_targets[valid_ignore] = -1

        # Force Match: Ensure every GT has at least one positive anchor (Best IoU)
        for i in range(len(gt_boxes)):
            gt = gt_boxes_np[i]
            gt_cls = int(gt_labels[i].item()) - 1

            dist_mask = dists[:, i] < dist_thresh
            cls_mask = anchor_class_indices == gt_cls
            candidate_mask = dist_mask & cls_mask
            candidate_idxs = torch.nonzero(candidate_mask, as_tuple=True)[0]

            if len(candidate_idxs) == 0:
                continue

            cand_anchors_np = anchors_np[candidate_idxs.cpu().numpy()]
            ious = [box_iou_3d_pair(ca, gt) for ca in cand_anchors_np]

            if not ious:
                continue

            ious = torch.tensor(ious, device=device)
            best_iou_idx = torch.argmax(ious)
            best_global_idx = candidate_idxs[best_iou_idx]

            # Force positive assignment
            cls_targets[best_global_idx] = 1

            gt_t = gt_boxes[i].unsqueeze(0)
            anc_t = anchors[best_global_idx].unsqueeze(0)
            reg_targets[best_global_idx] = box_encode(gt_t, anc_t).squeeze(0)

            anchor_yaw = anchors[best_global_idx, 6]
            gt_yaw = gt_boxes[i, 6]
            dir_targets[best_global_idx] = (torch.sin(gt_yaw - anchor_yaw) > 0).long()

        return cls_targets, reg_targets, dir_targets
