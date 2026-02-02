import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from library.config import Config
from library.utils import iou3d_shapely


class GaussianFocalLoss(nn.Module):
    """
    Focal Loss for Dense Object Detection (Heatmap).
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        pred = torch.clamp(pred, 1e-6, 1 - 1e-6)
        pos_inds = target.eq(1).float()
        neg_inds = target.lt(1).float()
        neg_weights = torch.pow(1 - target, 4)

        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

        num_pos = pos_inds.float().sum()
        if num_pos == 0:
            return -neg_loss.sum()
        return -(pos_loss.sum() + neg_loss.sum()) / num_pos


class TwoStageLoss(nn.Module):
    """
    Composite loss function for Two-Stage PointPillars.
    Stage 1: CenterHead Loss (Focal + L1)
    Stage 2: RoI Refinement Loss (L1) + IoU Rectification Loss (BCE)
    """

    def __init__(self):
        super().__init__()
        self.cls_weight = Config.LOSS_CLS_WEIGHT
        self.box_weight = Config.LOSS_BOX_WEIGHT
        self.iou_weight = Config.LOSS_IOU_WEIGHT
        self.focal_loss = GaussianFocalLoss()

    def gaussian_2d(self, shape, sigma=1):
        m, n = [(ss - 1.0) / 2.0 for ss in shape]
        y, x = np.ogrid[-m : m + 1, -n : n + 1]
        h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        return h

    def draw_gaussian(self, heatmap, center, radius, k=1):
        diameter = 2 * radius + 1
        gaussian = self.gaussian_2d((diameter, diameter), sigma=diameter / 6)
        x, y = int(center[0]), int(center[1])
        height, width = heatmap.shape[0:2]

        left, right = min(x, radius), min(width - x, radius + 1)
        top, bottom = min(y, radius), min(height - y, radius + 1)

        masked_heatmap = heatmap[y - top : y + bottom, x - left : x + right]
        masked_gaussian = gaussian[
            radius - top : radius + bottom, radius - left : radius + right
        ]

        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
            torch.max(
                masked_heatmap,
                torch.tensor(masked_gaussian, device=heatmap.device),
                out=masked_heatmap,
            )

    def compute_iou_cpu(self, boxes_a, boxes_b):
        a_np = boxes_a.detach().cpu().numpy()
        b_np = boxes_b.detach().cpu().numpy()
        iou = iou3d_shapely(a_np, b_np)
        return torch.from_numpy(iou)

    def compute_iou_diagonal_cpu(self, boxes_a, boxes_b):
        a_np = boxes_a.detach().cpu().numpy()
        b_np = boxes_b.detach().cpu().numpy()

        n = len(a_np)
        diag = np.zeros(n, dtype=np.float32)

        # Optimization: Loop and compute 1-to-1 to avoid O(N^2) complexity with Shapely
        for i in range(n):
            # iou3d_shapely expects (N, 7) inputs
            diag[i] = iou3d_shapely(a_np[i : i + 1], b_np[i : i + 1])[0, 0]

        return torch.from_numpy(diag)

    def forward(self, preds, batched_gt_boxes, batched_gt_labels):
        """
        Args:
            preds: Dict containing model outputs (stage1_preds, proposals, refined_boxes, pred_iou)
            batched_gt_boxes: List of (M, 7) tensors
            batched_gt_labels: List of (M,) tensors
        """
        device = preds["stage1_preds"][0]["hm"].device
        batch_size = len(batched_gt_boxes)
        stats = {}

        # ---------------------------------------------------------------------
        # Stage 1: CenterHead Loss
        # ---------------------------------------------------------------------
        stage1_loss = 0.0

        # Determine which tasks config to use
        # If preds length > Config.TASKS length, assume patched tasks (single class per task)
        tasks = Config.TASKS
        if len(preds["stage1_preds"]) > len(tasks):
            new_tasks = []
            for task in Config.TASKS:
                for class_name in task["class_names"]:
                    new_tasks.append(dict(num_class=1, class_names=[class_name]))
            tasks = new_tasks

        feature_map = preds["stage1_preds"][0]["hm"]
        B, _, H, W = feature_map.shape

        for task_idx, task in enumerate(tasks):
            if task_idx >= len(preds["stage1_preds"]):
                break

            task_preds = preds["stage1_preds"][task_idx]

            # Prepare Targets
            target_hm = torch.zeros_like(task_preds["hm"])
            target_reg = torch.zeros_like(task_preds["reg"])
            target_height = torch.zeros_like(task_preds["height"])
            target_dim = torch.zeros_like(task_preds["dim"])
            target_rot = torch.zeros_like(task_preds["rot"])
            reg_mask = torch.zeros((B, 1, H, W), device=device)

            for b in range(B):
                gt_boxes = batched_gt_boxes[b]
                gt_labels = batched_gt_labels[b]

                if len(gt_boxes) == 0:
                    continue

                # Filter GT for this task
                task_class_names = task["class_names"]
                task_class_indices = [
                    Config.CLASS_NAMES.index(n) for n in task_class_names
                ]

                # Create mask for boxes belonging to this task
                mask = torch.zeros_like(gt_labels, dtype=torch.bool)
                for cls_idx in task_class_indices:
                    mask = mask | (gt_labels == cls_idx)

                task_boxes = gt_boxes[mask]
                task_labels = gt_labels[mask]

                if len(task_boxes) == 0:
                    continue

                # Map to Grid
                xs = task_boxes[:, 0]
                ys = task_boxes[:, 1]

                x_idx = (
                    (xs - Config.POINT_CLOUD_RANGE[0]) / Config.VOXEL_SIZE[0]
                ).long()
                y_idx = (
                    (ys - Config.POINT_CLOUD_RANGE[1]) / Config.VOXEL_SIZE[1]
                ).long()

                x_idx = torch.clamp(x_idx, 0, W - 1)
                y_idx = torch.clamp(y_idx, 0, H - 1)

                for k in range(len(task_boxes)):
                    cx, cy = x_idx[k], y_idx[k]

                    # Determine class channel within the task head
                    cls_global = task_labels[k].item()
                    cls_name = Config.CLASS_NAMES[cls_global]
                    cls_local = task_class_names.index(cls_name)

                    # Heatmap
                    self.draw_gaussian(target_hm[b, cls_local], (cx, cy), radius=2)

                    # Regression Mask
                    reg_mask[b, 0, cy, cx] = 1

                    # Regression Offsets
                    real_cx = (xs[k] - Config.POINT_CLOUD_RANGE[0]) / Config.VOXEL_SIZE[
                        0
                    ]
                    real_cy = (ys[k] - Config.POINT_CLOUD_RANGE[1]) / Config.VOXEL_SIZE[
                        1
                    ]

                    target_reg[b, 0, cy, cx] = real_cx - cx.float()
                    target_reg[b, 1, cy, cx] = real_cy - cy.float()

                    # Height
                    target_height[b, 0, cy, cx] = task_boxes[k, 2]

                    # Dim (Log)
                    target_dim[b, 0, cy, cx] = torch.log(task_boxes[k, 3])
                    target_dim[b, 1, cy, cx] = torch.log(task_boxes[k, 4])
                    target_dim[b, 2, cy, cx] = torch.log(task_boxes[k, 5])

                    # Rot (Sin, Cos)
                    target_rot[b, 0, cy, cx] = torch.sin(task_boxes[k, 6])
                    target_rot[b, 1, cy, cx] = torch.cos(task_boxes[k, 6])

            # Compute Loss
            hm_loss = self.focal_loss(task_preds["hm"], target_hm)

            num_pos = reg_mask.sum()
            num_pos = torch.clamp(num_pos, min=1.0)

            def masked_l1(p, t, m):
                return (torch.abs(p - t) * m).sum() / num_pos

            reg_loss = masked_l1(task_preds["reg"], target_reg, reg_mask)
            h_loss = masked_l1(task_preds["height"], target_height, reg_mask)
            d_loss = masked_l1(task_preds["dim"], target_dim, reg_mask)
            r_loss = masked_l1(task_preds["rot"], target_rot, reg_mask)

            task_loss = (
                hm_loss * self.cls_weight
                + (reg_loss + h_loss + d_loss + r_loss) * self.box_weight
            )
            stage1_loss += task_loss

        stats["stage1_loss"] = stage1_loss.item()

        # ---------------------------------------------------------------------
        # Stage 2: Refinement & IoU Loss
        # ---------------------------------------------------------------------
        stage2_loss = torch.tensor(0.0, device=device)
        proposals = preds["proposals"]

        if proposals is not None and proposals.shape[0] > 0:
            refined_boxes = preds["refined_boxes"]
            pred_iou = preds["pred_iou"]
            batch_idx = proposals[:, 8].long()
            unique_batches = torch.unique(batch_idx)

            total_refine_loss = 0.0
            total_iou_loss = 0.0
            total_samples = 0

            for b in unique_batches:
                mask = batch_idx == b
                b_props = proposals[mask]
                b_refined = refined_boxes[mask]
                b_pred_iou = pred_iou[mask]
                b_gt = batched_gt_boxes[b]

                if len(b_gt) == 0:
                    target_iou = torch.zeros_like(b_pred_iou)
                    total_iou_loss += F.binary_cross_entropy(
                        b_pred_iou, target_iou, reduction="sum"
                    )
                    total_samples += len(b_props)
                    continue

                # Match Proposals to GT
                iou_matrix = self.compute_iou_cpu(b_props[:, :7], b_gt)
                iou_matrix = iou_matrix.to(device)

                max_ious, gt_indices = torch.max(iou_matrix, dim=1)

                # 1. Refinement Loss (Positives only: IoU > 0.5)
                pos_mask = max_ious > 0.5

                if pos_mask.sum() > 0:
                    pos_refined = b_refined[pos_mask]
                    pos_gt = b_gt[gt_indices[pos_mask]]

                    # Delta X, Y, Z, Yaw
                    loss_loc = F.l1_loss(
                        pos_refined[:, [0, 1, 2, 6]],
                        pos_gt[:, [0, 1, 2, 6]],
                        reduction="sum",
                    )

                    # Delta Dim (Log space)
                    pred_dim = torch.log(pos_refined[:, 3:6])
                    target_dim = torch.log(pos_gt[:, 3:6])
                    loss_dim = F.l1_loss(pred_dim, target_dim, reduction="sum")

                    total_refine_loss += loss_loc + loss_dim

                # 2. IoU Loss (All proposals)
                matched_gt = b_gt[gt_indices]
                target_iou_vals = self.compute_iou_diagonal_cpu(b_refined, matched_gt)
                target_iou_vals = target_iou_vals.to(device).unsqueeze(1)
                target_iou_vals = torch.clamp(target_iou_vals, 0.0, 1.0)

                total_iou_loss += F.binary_cross_entropy(
                    b_pred_iou, target_iou_vals, reduction="sum"
                )
                total_samples += len(b_props)

            if total_samples > 0:
                stage2_loss = (
                    total_refine_loss * self.box_weight
                    + total_iou_loss * self.iou_weight
                ) / total_samples

        stats["stage2_loss"] = stage2_loss.item()
        total_loss = stage1_loss + stage2_loss

        return total_loss, stats
