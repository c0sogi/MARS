import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os

# Import library components
from library.config import Config
from library.modules import PointPillars
from library.utils import iou3d_shapely, nms_3d

# -----------------------------------------------------------------------------
# RUNTIME PATCH: Fix Critical Bug in Library Code
# -----------------------------------------------------------------------------
# The provided CenterHead.get_proposals method incorrectly calculates coordinates
# for tasks with num_classes > 1 (ys = inds // W overflows if channels > 1).
# We patch Config.TASKS to ensure every task has exactly 1 class.
# This must be done BEFORE instantiating the model.
# -----------------------------------------------------------------------------
NEW_TASKS = []
for task in Config.TASKS:
    for class_name in task["class_names"]:
        NEW_TASKS.append(dict(num_class=1, class_names=[class_name]))
Config.TASKS = NEW_TASKS


class TwoStagePointPillars(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = PointPillars()

        # Loss Weights
        self.cls_weight = Config.LOSS_CLS_WEIGHT
        self.box_weight = Config.LOSS_BOX_WEIGHT
        self.iou_weight = Config.LOSS_IOU_WEIGHT

        # Cache for Gaussian generation
        self.gaussian_cache = {}

    def forward(self, batched_points, batched_gt_boxes=None, batched_gt_labels=None):
        """
        Args:
            batched_points: List of (N, 4) tensors.
            batched_gt_boxes: List of (M, 7) tensors (Train only).
            batched_gt_labels: List of (M,) tensors (Train only).
        """
        # 1. Forward Pass
        preds = self.model(batched_points)

        # 2. Training Logic
        if self.training:
            if batched_gt_boxes is None:
                raise ValueError("GT boxes required for training")
            loss, stats = self.compute_loss(preds, batched_gt_boxes, batched_gt_labels)
            return loss, stats

        # 3. Inference Logic
        else:
            return self.post_process(preds)

    def compute_loss(self, preds, batched_gt_boxes, batched_gt_labels):
        device = preds["stage1_preds"][0]["hm"].device
        batch_size = len(batched_gt_boxes)

        total_loss = 0.0
        stats = {}

        # ---------------------------------------------------------------------
        # Stage 1: CenterHead Loss (Focal + L1)
        # ---------------------------------------------------------------------
        stage1_loss = 0.0

        # Grid dimensions from feature map
        feature_map = preds["stage1_preds"][0]["hm"]
        B, _, H, W = feature_map.shape

        for task_idx, task in enumerate(Config.TASKS):
            task_preds = preds["stage1_preds"][task_idx]

            # Prepare Targets
            target_hm = torch.zeros_like(task_preds["hm"])
            target_reg = torch.zeros_like(task_preds["reg"])
            target_height = torch.zeros_like(task_preds["height"])
            target_dim = torch.zeros_like(task_preds["dim"])
            target_rot = torch.zeros_like(task_preds["rot"])
            reg_mask = torch.zeros((B, 1, H, W), device=device)

            # Fill Targets
            for b in range(B):
                gt_boxes = batched_gt_boxes[b]
                gt_labels = batched_gt_labels[b]

                if len(gt_boxes) == 0:
                    continue

                # Filter GT for this task
                # Since we patched tasks to be single-class, logic is simple
                class_name = task["class_names"][0]
                class_idx = Config.CLASS_NAMES.index(class_name)

                mask = gt_labels == class_idx
                task_boxes = gt_boxes[mask]

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

                # Draw Targets
                for k in range(len(task_boxes)):
                    cx, cy = x_idx[k], y_idx[k]

                    # Heatmap (Gaussian)
                    self.draw_gaussian(target_hm[b, 0], (cx, cy), radius=2)

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

            # Masked L1 Loss
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
        stage2_loss = 0.0
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
                    # No GT: All negatives. Target IoU = 0
                    target_iou = torch.zeros_like(b_pred_iou)
                    total_iou_loss += F.binary_cross_entropy(
                        b_pred_iou, target_iou, reduction="sum"
                    )
                    total_samples += len(b_props)
                    continue

                # Match Proposals to GT (using CPU Shapely for correctness)
                # We use ORIGINAL proposals for matching
                iou_matrix = self.compute_iou_cpu(
                    b_props[:, :7], b_gt
                )  # (N_prop, N_gt)
                iou_matrix = iou_matrix.to(device)

                max_ious, gt_indices = torch.max(iou_matrix, dim=1)

                # 1. Refinement Loss (Positives only: IoU > 0.5)
                pos_mask = max_ious > 0.5
                if pos_mask.sum() > 0:
                    pos_refined = b_refined[pos_mask]
                    pos_props = b_props[pos_mask]
                    pos_gt = b_gt[gt_indices[pos_mask]]

                    # Calculate Targets (Residuals)
                    # We compare inferred residuals from refined boxes to target residuals
                    # Target: GT - Proposal
                    # Pred: Refined - Proposal

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
                # Target is IoU(Refined_Box, Matched_GT)
                # We calculate this on CPU to be safe with rotation
                matched_gt = b_gt[gt_indices]

                # Compute actual IoU of refined boxes
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

        stats["stage2_loss"] = (
            stage2_loss.item() if isinstance(stage2_loss, torch.Tensor) else stage2_loss
        )

        total_loss = stage1_loss + stage2_loss
        return total_loss, stats

    def post_process(self, preds):
        """
        Reconstruct class labels and format predictions.
        """
        if preds["refined_boxes"] is None:
            return []

        refined_boxes = preds["refined_boxes"]  # (N, 7)
        scores = preds["rectified_scores"]  # (N,)
        proposals = preds["proposals"]  # (N, 9)

        # Recover Class IDs
        # Since we patched TASKS to be single-class, we can recover class ID
        # by re-running the proposal selection logic to see which task generated which box.
        # However, PointPillars output is concatenated.
        # We need to re-generate proposals with class info to match them up.

        # Re-run get_proposals logic locally
        preds_dicts = preds["stage1_preds"]
        class_labels = []

        # We must iterate exactly as PointPillars.get_proposals does
        for task_idx, task_preds in enumerate(preds_dicts):
            hm = torch.sigmoid(task_preds["hm"])
            B, _, H, W = hm.shape

            # NMS
            pad = 1
            hmax = F.max_pool2d(hm, kernel_size=3, stride=1, padding=pad)
            keep = (hmax == hm).float()
            hm = hm * keep

            hm = hm.view(B, -1)
            K = min(Config.MAX_PROPOSALS, hm.shape[1])
            task_scores, _ = torch.topk(hm, K)

            # We just need the count of valid proposals per batch per task
            for b in range(B):
                mask = task_scores[b] > Config.SCORE_THRESHOLD
                count = mask.sum().item()
                # Append class ID 'count' times
                # task_idx corresponds to Config.TASKS[task_idx]
                # Since we patched tasks, task_idx IS the class index in Config.TASKS order
                # But Config.TASKS order might differ from Config.CLASS_NAMES?
                # We constructed NEW_TASKS by iterating Config.TASKS.
                # Let's map task_idx back to class name.
                class_name = Config.TASKS[task_idx]["class_names"][0]
                class_labels.extend([class_name] * count)

        # Verify alignment
        if len(class_labels) != len(refined_boxes):
            # Fallback if mismatch (should not happen if logic is identical)
            # Return empty or best guess
            return []

        # Format for Submission
        # Need: [x, y, z, w, l, h, yaw, score, class_name]
        # Group by batch index
        batch_idx = proposals[:, 8].long().cpu().numpy()
        boxes_np = refined_boxes.detach().cpu().numpy()
        scores_np = scores.detach().cpu().numpy()

        formatted_preds = []
        unique_batches = np.unique(batch_idx)

        # Map flat list back to batch structure
        # class_labels is flat list matching 'proposals' order

        current_idx = 0

        # We need to reconstruct the batch grouping from the flat class_labels list
        # The proposals tensor is grouped by Task then Batch?
        # No, PointPillars.get_proposals:
        # for task: for batch: append()
        # So it is: T0_B0, T0_B1, T1_B0, T1_B1...
        # But `torch.cat` flattens it.

        # Wait, `refined_boxes` corresponds to `proposals`.
        # `proposals` is `torch.cat(proposals_list, dim=0)`.
        # `proposals_list` was appended in the loop order.
        # So `class_labels` (which we built in loop order) aligns 1-to-1 with `refined_boxes`.

        # Now we need to group by batch for the output format (list of strings per sample)
        # But `batch_idx` in proposals tells us which sample it belongs to.

        # Create a list of lists [Batch0_Preds, Batch1_Preds...]
        # We assume batch size is known or max(batch_idx)

        # Initialize dict
        batch_preds = {}

        for i in range(len(boxes_np)):
            b = batch_idx[i]
            if b not in batch_preds:
                batch_preds[b] = []

            # Apply NMS per sample later?
            # The model output is raw proposals. We need Final NMS.

            pred = {"box": boxes_np[i], "score": scores_np[i], "class": class_labels[i]}
            batch_preds[b].append(pred)

        final_output = []
        # Iterate over batch size (we need to return list of length B)
        # We don't know B easily here, but we can infer from max batch_idx
        # Or pass B in. Assuming B matches input.

        # Just return a list of prediction strings/dicts keyed by batch index
        # The caller (training loop) will handle mapping to sample tokens.

        sorted_batches = sorted(batch_preds.keys())
        # If a batch has no preds, it won't be in keys.
        # We return a dict {batch_idx: "pred_string"}

        results = {}
        for b, preds_list in batch_preds.items():
            # NMS
            boxes = np.array([p["box"] for p in preds_list])
            scores = np.array([p["score"] for p in preds_list])
            classes = np.array([p["class"] for p in preds_list])

            keep = nms_3d(boxes, scores, iou_threshold=Config.NMS_IOU_THRESHOLD)

            # Format String
            # score x y z w l h yaw class
            pred_strings = []
            for k in keep:
                p_box = boxes[k]
                p_score = scores[k]
                p_cls = classes[k]

                # Format: score x y z w l h yaw class
                s = f"{p_score:.4f} {p_box[0]:.4f} {p_box[1]:.4f} {p_box[2]:.4f} {p_box[3]:.4f} {p_box[4]:.4f} {p_box[5]:.4f} {p_box[6]:.4f} {p_cls}"
                pred_strings.append(s)

            results[b] = " ".join(pred_strings)

        return results

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------
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

    def focal_loss(self, pred, target):
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

    def compute_iou_cpu(self, boxes_a, boxes_b):
        a_np = boxes_a.detach().cpu().numpy()
        b_np = boxes_b.detach().cpu().numpy()
        iou = iou3d_shapely(a_np, b_np)
        return torch.from_numpy(iou)

    def compute_iou_diagonal_cpu(self, boxes_a, boxes_b):
        # Compute pairwise IoU for corresponding indices
        a_np = boxes_a.detach().cpu().numpy()
        b_np = boxes_b.detach().cpu().numpy()

        n = len(a_np)
        diag = np.zeros(n, dtype=np.float32)

        # Optimization: Loop and compute 1-to-1 to avoid O(N^2) complexity with Shapely
        for i in range(n):
            # iou3d_shapely expects (N, 7) inputs
            diag[i] = iou3d_shapely(a_np[i : i + 1], b_np[i : i + 1])[0, 0]

        return torch.from_numpy(diag)
