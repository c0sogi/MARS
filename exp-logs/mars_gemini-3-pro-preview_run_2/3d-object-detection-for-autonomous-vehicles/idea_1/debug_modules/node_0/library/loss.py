import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config
from library.anchors import AnchorGenerator


class LossModule(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config if config is not None else Config
        self.anchor_generator = AnchorGenerator(self.config)
        self.class_names = self.config.CLASS_NAMES
        self.anchor_rotations = self.config.ANCHOR_ROTATIONS

        # Hyperparameters
        self.alpha = 0.25
        self.gamma = 2.0
        self.cls_weight = 1.0
        self.reg_weight = 2.0

    def forward(self, cls_preds, reg_preds, batch_gt_boxes, batch_gt_labels):
        """
        Args:
            cls_preds: (B, H, W, Num_Anchors, Num_Classes)
            reg_preds: (B, H, W, Num_Anchors, 7)
            batch_gt_boxes: List of Tensors [(M, 7), ...]
            batch_gt_labels: List of Tensors [(M,), ...]
        """
        batch_size = cls_preds.shape[0]
        device = cls_preds.device

        # 1. Generate Anchors
        # Shape: (H, W, Num_Anchors, 7)
        # Note: Num_Anchors here refers to Num_Types (Classes * Rotations)
        feature_map_size = (cls_preds.shape[1], cls_preds.shape[2])
        anchors = self.anchor_generator.generate(feature_map_size, device=device)

        # Flatten anchors: (N_a, 7)
        # Reshape order matches the prediction flattening order (H, W, A)
        anchors = anchors.reshape(-1, 7)
        num_anchors = anchors.shape[0]

        # 2. Prepare Targets
        batch_cls_targets = []
        batch_reg_targets = []
        batch_cls_weights = []
        batch_reg_weights = []

        # Helper to map anchor index to class index
        num_rots = len(self.anchor_rotations)
        num_types = len(self.class_names) * num_rots

        # Create a mapping of anchor_index -> class_index
        # Anchors are generated in loops: Class -> Rotation -> Grid
        # But the output of generate() is (H, W, Types, 7)
        # So when flattened to (-1, 7), the inner dimension is Types.
        # Index i corresponds to type (i % num_types)
        anchor_indices = torch.arange(num_anchors, device=device)
        anchor_type_indices = anchor_indices % num_types
        anchor_class_indices = torch.div(
            anchor_type_indices, num_rots, rounding_mode="floor"
        )

        for b in range(batch_size):
            gt_boxes = batch_gt_boxes[b].to(device)
            gt_labels = batch_gt_labels[b].to(
                device
            )  # 1-based index from config mapping

            # Initialize targets
            # cls_target: 1 for positive class, 0 otherwise
            cls_target = torch.zeros(
                (num_anchors, len(self.class_names)), device=device
            )
            reg_target = torch.zeros((num_anchors, 7), device=device)

            # weights: 1 for valid pos/neg, 0 for ignore
            cls_weight = torch.zeros((num_anchors,), device=device)
            reg_weight = torch.zeros((num_anchors,), device=device)

            if len(gt_boxes) > 0:
                # Process per class to enforce type matching
                for class_idx, class_name in enumerate(self.class_names):
                    # 1. Select anchors for this class
                    class_anchor_mask = anchor_class_indices == class_idx
                    relevant_anchor_indices = torch.where(class_anchor_mask)[0]
                    relevant_anchors = anchors[class_anchor_mask]

                    if relevant_anchors.shape[0] == 0:
                        continue

                    # 2. Select GT for this class
                    # gt_labels are 1-based, so class_idx + 1 matches
                    class_gt_mask = gt_labels == (class_idx + 1)
                    class_gt_boxes = gt_boxes[class_gt_mask]

                    # Get Thresholds
                    pos_thresh, neg_thresh = self.config.ANCHOR_MATCH_THRESHOLDS.get(
                        class_name,
                        self.config.ANCHOR_MATCH_THRESHOLDS.get("default", (0.5, 0.35)),
                    )

                    if class_gt_boxes.shape[0] == 0:
                        # No GT for this class -> All anchors are negative
                        cls_weight[relevant_anchor_indices] = 1.0
                        continue

                    # 3. Compute IoU (Axis Aligned)
                    # (N_rel_anchors, N_class_gt)
                    ious = self._compute_iou_torch(relevant_anchors, class_gt_boxes)

                    # 4. Match
                    max_ious, max_ids = torch.max(ious, dim=1)

                    # Positives
                    pos_mask = max_ious >= pos_thresh
                    # Negatives
                    neg_mask = max_ious < neg_thresh

                    # Global Indices
                    pos_global_ids = relevant_anchor_indices[pos_mask]
                    neg_global_ids = relevant_anchor_indices[neg_mask]

                    # 5. Assign Classification Targets
                    # Positives: Set specific class bit to 1, weight 1
                    cls_target[pos_global_ids, class_idx] = 1.0
                    cls_weight[pos_global_ids] = 1.0

                    # Negatives: Target 0, weight 1
                    cls_weight[neg_global_ids] = 1.0

                    # Ignored anchors (between thresholds) have weight 0 (default)

                    # 6. Assign Regression Targets (Positives only)
                    if pos_mask.any():
                        matched_gt = class_gt_boxes[max_ids[pos_mask]]
                        matched_anchors = relevant_anchors[pos_mask]

                        # Encode Targets
                        # Anchor dims
                        ma_dims = matched_anchors[:, 3:6]
                        ma_d = torch.sqrt(ma_dims[:, 0] ** 2 + ma_dims[:, 1] ** 2)
                        ma_h = ma_dims[:, 2]
                        ma_center = matched_anchors[:, :3]
                        ma_yaw = matched_anchors[:, 6]

                        # GT dims
                        mg_dims = matched_gt[:, 3:6]
                        mg_center = matched_gt[:, :3]
                        mg_yaw = matched_gt[:, 6]

                        # Encoding
                        dx = (mg_center[:, 0] - ma_center[:, 0]) / ma_d
                        dy = (mg_center[:, 1] - ma_center[:, 1]) / ma_d
                        dz = (mg_center[:, 2] - ma_center[:, 2]) / ma_h
                        dw = torch.log(mg_dims[:, 0] / ma_dims[:, 0])
                        dl = torch.log(mg_dims[:, 1] / ma_dims[:, 1])
                        dh = torch.log(mg_dims[:, 2] / ma_dims[:, 2])
                        dyaw = torch.sin(mg_yaw - ma_yaw)

                        encoded_targets = torch.stack(
                            [dx, dy, dz, dw, dl, dh, dyaw], dim=1
                        )

                        reg_target[pos_global_ids] = encoded_targets
                        reg_weight[pos_global_ids] = 1.0
            else:
                # No GT in sample, all anchors are negative
                cls_weight[:] = 1.0

            batch_cls_targets.append(cls_target)
            batch_reg_targets.append(reg_target)
            batch_cls_weights.append(cls_weight)
            batch_reg_weights.append(reg_weight)

        # Stack batches
        # (B, N_a, C)
        batch_cls_targets = torch.stack(batch_cls_targets)
        batch_reg_targets = torch.stack(batch_reg_targets)
        batch_cls_weights = torch.stack(batch_cls_weights)
        batch_reg_weights = torch.stack(batch_reg_weights)

        # Flatten for loss
        flat_cls_preds = cls_preds.reshape(-1, len(self.class_names))
        flat_reg_preds = reg_preds.reshape(-1, 7)

        flat_cls_targets = batch_cls_targets.reshape(-1, len(self.class_names))
        flat_reg_targets = batch_reg_targets.reshape(-1, 7)
        flat_cls_weights = batch_cls_weights.reshape(-1)
        flat_reg_weights = batch_reg_weights.reshape(-1)

        # --- Compute Losses ---

        # 1. Classification Loss (Sigmoid Focal Loss)
        probs = torch.sigmoid(flat_cls_preds)

        # pt: probability of the ground truth class
        pt = torch.where(flat_cls_targets == 1, probs, 1 - probs)
        # alpha_t: alpha for class 1, 1-alpha for class 0
        alpha_t = torch.where(flat_cls_targets == 1, self.alpha, 1 - self.alpha)

        focal_loss = -alpha_t * (1 - pt).pow(self.gamma) * torch.log(pt + 1e-6)

        # Normalization
        num_positives = flat_reg_weights.sum()
        normalizer = torch.clamp(num_positives, min=1.0)

        # Apply weights (handles ignore mask)
        # weight is (N,), loss is (N, C). Broadcast.
        cls_loss = (focal_loss * flat_cls_weights.unsqueeze(1)).sum() / normalizer

        # 2. Regression Loss (Smooth L1)
        reg_loss_item = F.smooth_l1_loss(
            flat_reg_preds, flat_reg_targets, reduction="none"
        )
        reg_loss = (reg_loss_item * flat_reg_weights.unsqueeze(1)).sum() / normalizer

        total_loss = cls_loss * self.cls_weight + reg_loss * self.reg_weight

        return total_loss, {"cls_loss": cls_loss.item(), "reg_loss": reg_loss.item()}

    def _compute_iou_torch(self, anchors, gt_boxes):
        """
        Compute Axis-Aligned 3D IoU between anchors and GT boxes using PyTorch.
        Args:
            anchors: (N, 7)
            gt_boxes: (M, 7)
        Returns:
            ious: (N, M)
        """
        # Expand for broadcasting: (N, 1, 7) vs (1, M, 7)
        anchors = anchors.unsqueeze(1)
        gt = gt_boxes.unsqueeze(0)

        # Extract dims
        # Anchors: [x, y, z, w, l, h, yaw]
        # In NuScenes/Config: w is Y-size, l is X-size
        a_x, a_y, a_z = anchors[..., 0], anchors[..., 1], anchors[..., 2]
        a_w, a_l, a_h = anchors[..., 3], anchors[..., 4], anchors[..., 5]

        g_x, g_y, g_z = gt[..., 0], gt[..., 1], gt[..., 2]
        g_w, g_l, g_h = gt[..., 3], gt[..., 4], gt[..., 5]

        # Intersection
        # Width (Y-axis)
        iw_min = torch.max(a_y - a_w / 2, g_y - g_w / 2)
        iw_max = torch.min(a_y + a_w / 2, g_y + g_w / 2)
        iw = torch.clamp(iw_max - iw_min, min=0)

        # Length (X-axis)
        il_min = torch.max(a_x - a_l / 2, g_x - g_l / 2)
        il_max = torch.min(a_x + a_l / 2, g_x + g_l / 2)
        il = torch.clamp(il_max - il_min, min=0)

        # Height (Z-axis)
        ih_min = torch.max(a_z - a_h / 2, g_z - g_h / 2)
        ih_max = torch.min(a_z + a_h / 2, g_z + g_h / 2)
        ih = torch.clamp(ih_max - ih_min, min=0)

        inter_vol = iw * il * ih

        vol_a = a_w * a_l * a_h
        vol_g = g_w * g_l * g_h

        union_vol = vol_a + vol_g - inter_vol

        return inter_vol / (union_vol + 1e-6)
