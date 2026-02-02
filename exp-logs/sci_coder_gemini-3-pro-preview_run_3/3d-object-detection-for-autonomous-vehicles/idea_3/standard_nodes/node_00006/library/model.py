import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from library.config import (
    POINT_CLOUD_RANGE,
    VOXEL_SIZE,
    GRID_SIZE,
    NUM_POINT_FEATURES,
    NUM_PILLAR_FEATURES,
    LAYER_STRIDES,
    LAYER_FILTERS,
    UPSAMPLE_STRIDES,
    NUM_UPSAMPLE_FILTERS,
    ANCHOR_CONFIGS,
    CLASS_NAMES,
    LOSS_WEIGHTS,
    SCORE_THRESHOLD,
    NMS_IOU_THRESHOLD,
    MAX_DETECTIONS,
)
from library.utils import (
    box_encode,
    box_decode,
    nms_3d,
    box_iou_3d_pair,
    get_corners_2d,
)

# ==============================================================================
# MODULES
# ==============================================================================


class PillarFeatureNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_input_features = NUM_POINT_FEATURES + 5  # x,y,z,i + xc,yc,zc,xp,yp
        self.num_output_features = NUM_PILLAR_FEATURES

        # Simplified PointNet
        self.linear = nn.Linear(self.num_input_features, self.num_output_features)
        self.norm = nn.BatchNorm1d(self.num_output_features)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, features, num_points, coords):
        """
        Args:
            features: (P, N, 4) [x, y, z, i]
            num_points: (P,)
            coords: (P, 4) [b, z, y, x]
        Returns:
            pillar_features: (P, 64)
        """
        # Calculate offsets
        # 1. Arithmetic Mean (xc, yc, zc)
        # Sum over points and divide by num_points
        # Mask out zero-padded points for mean calculation
        dtype = features.dtype
        device = features.device

        # Create mask for valid points
        # features shape: (P, N, 4)
        P, N, _ = features.shape

        # Create a mask based on num_points
        # indices: (1, N)
        indices = torch.arange(N, device=device).unsqueeze(0)
        # mask: (P, N)
        mask = indices < num_points.unsqueeze(1)
        mask = mask.unsqueeze(2)  # (P, N, 1)

        # Sum valid points
        masked_features = features * mask
        sum_points = masked_features.sum(dim=1)  # (P, 4)

        # Mean
        # Avoid division by zero
        num_points_safe = torch.clamp(num_points, min=1).view(-1, 1).type(dtype)
        mean_points = sum_points[:, :3] / num_points_safe  # (P, 3)

        # Expand mean to (P, N, 3)
        mean_expanded = mean_points.unsqueeze(1).expand(-1, N, -1)

        # Offset from mean
        offset_mean = features[..., :3] - mean_expanded

        # 2. Geometric Center (xp, yp)
        # Calculate center of the pillar based on coords
        # coords: [b, z, y, x]
        # x_center = x_idx * v_x + min_x + v_x/2
        x_idx = coords[:, 3].type(dtype)
        y_idx = coords[:, 2].type(dtype)

        x_center = x_idx * VOXEL_SIZE[0] + POINT_CLOUD_RANGE[0] + VOXEL_SIZE[0] / 2
        y_center = y_idx * VOXEL_SIZE[1] + POINT_CLOUD_RANGE[1] + VOXEL_SIZE[1] / 2

        # Expand
        center_expanded = (
            torch.stack([x_center, y_center], dim=1).unsqueeze(1).expand(-1, N, -1)
        )

        # Offset from geometric center
        offset_center = features[..., :2] - center_expanded

        # Concatenate all features
        # [x, y, z, i, x-xc, y-yc, z-zc, x-xp, y-yp]
        input_features = torch.cat([features, offset_mean, offset_center], dim=2)

        # Forward PointNet
        # Flatten (P*N, C)
        x = input_features.view(-1, self.num_input_features)
        x = self.linear(x)
        x = self.norm(x)
        x = self.relu(x)

        # Reshape back (P, N, C)
        x = x.view(P, N, self.num_output_features)

        # Max Pooling over points
        # Apply mask to ensure padded points don't affect max (set to -inf)
        # But ReLU makes everything >= 0. So 0 padding is fine if we max?
        # If all points are 0 (padded), max is 0.
        # However, we want max over *valid* points.
        # Since we use ReLU, valid features are >= 0.
        # If we just take max, 0s from padding might be selected if valid points are all 0?
        # But valid points usually have some activation.
        # To be safe, we can mask the output.
        x = x * mask  # Zero out invalid points again

        x_max = torch.max(x, dim=1)[0]  # (P, 64)

        return x_max


class PointPillarsScatter(nn.Module):
    def __init__(self):
        super().__init__()
        self.nx = GRID_SIZE[0]
        self.ny = GRID_SIZE[1]
        self.num_channels = NUM_PILLAR_FEATURES

    def forward(self, pillar_features, coords, batch_size):
        """
        Args:
            pillar_features: (P, 64)
            coords: (P, 4) [b, z, y, x]
            batch_size: int
        Returns:
            batch_canvas: (B, 64, H, W)
        """
        device = pillar_features.device

        # Create canvas
        canvas = torch.zeros(
            (batch_size, self.num_channels, self.ny * self.nx),
            dtype=pillar_features.dtype,
            device=device,
        )

        # Unpack coords
        b_idx = coords[:, 0].long()
        y_idx = coords[:, 2].long()
        x_idx = coords[:, 3].long()

        # Calculate linear index
        indices = b_idx * (self.ny * self.nx) + y_idx * self.nx + x_idx

        # Scatter
        # Transpose features to (C, P) for simpler indexing?
        # Actually we need to place (P, C) into (B, C, H*W)
        # We can reshape canvas to (B*H*W, C) -> scatter -> reshape

        canvas_flat = canvas.view(-1, self.num_channels)  # (B*H*W, C)

        # We need to assign pillar_features to canvas_flat[indices]
        # Since indices might have duplicates (unlikely with unique pillars), simple assignment works
        # But scatter is safer

        # canvas_flat[indices] = pillar_features
        # This works if indices are unique. In PointPillars, pillars are unique grid cells.
        canvas_flat.index_add_(0, indices, pillar_features)

        # Reshape back
        batch_canvas = canvas_flat.view(batch_size, self.ny, self.nx, self.num_channels)
        # Permute to (B, C, H, W)
        batch_canvas = batch_canvas.permute(0, 3, 1, 2).contiguous()

        return batch_canvas


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()

        # Downsampling Blocks
        # Block 1
        self.block1 = nn.Sequential(
            nn.Conv2d(
                NUM_PILLAR_FEATURES,
                LAYER_FILTERS[0],
                3,
                stride=LAYER_STRIDES[0],
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(LAYER_FILTERS[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(LAYER_FILTERS[0], LAYER_FILTERS[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(LAYER_FILTERS[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(LAYER_FILTERS[0], LAYER_FILTERS[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(LAYER_FILTERS[0]),
            nn.ReLU(inplace=True),
        )

        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(
                LAYER_FILTERS[0],
                LAYER_FILTERS[1],
                3,
                stride=LAYER_STRIDES[1],
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(LAYER_FILTERS[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(LAYER_FILTERS[1], LAYER_FILTERS[1], 3, padding=1, bias=False),
            nn.BatchNorm2d(LAYER_FILTERS[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(LAYER_FILTERS[1], LAYER_FILTERS[1], 3, padding=1, bias=False),
            nn.BatchNorm2d(LAYER_FILTERS[1]),
            nn.ReLU(inplace=True),
        )

        # Block 3
        self.block3 = nn.Sequential(
            nn.Conv2d(
                LAYER_FILTERS[1],
                LAYER_FILTERS[2],
                3,
                stride=LAYER_STRIDES[2],
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(LAYER_FILTERS[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(LAYER_FILTERS[2], LAYER_FILTERS[2], 3, padding=1, bias=False),
            nn.BatchNorm2d(LAYER_FILTERS[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(LAYER_FILTERS[2], LAYER_FILTERS[2], 3, padding=1, bias=False),
            nn.BatchNorm2d(LAYER_FILTERS[2]),
            nn.ReLU(inplace=True),
        )

        # Upsampling Blocks
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(
                LAYER_FILTERS[0],
                NUM_UPSAMPLE_FILTERS[0],
                UPSAMPLE_STRIDES[0],
                stride=UPSAMPLE_STRIDES[0],
                bias=False,
            ),
            nn.BatchNorm2d(NUM_UPSAMPLE_FILTERS[0]),
            nn.ReLU(inplace=True),
        )

        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(
                LAYER_FILTERS[1],
                NUM_UPSAMPLE_FILTERS[1],
                UPSAMPLE_STRIDES[1],
                stride=UPSAMPLE_STRIDES[1],
                bias=False,
            ),
            nn.BatchNorm2d(NUM_UPSAMPLE_FILTERS[1]),
            nn.ReLU(inplace=True),
        )

        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(
                LAYER_FILTERS[2],
                NUM_UPSAMPLE_FILTERS[2],
                UPSAMPLE_STRIDES[2],
                stride=UPSAMPLE_STRIDES[2],
                bias=False,
            ),
            nn.BatchNorm2d(NUM_UPSAMPLE_FILTERS[2]),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)

        u1 = self.deconv1(x1)
        u2 = self.deconv2(x2)
        u3 = self.deconv3(x3)

        # Concatenate
        # Ensure sizes match (sometimes padding causes 1px diff)
        # Assuming input is multiple of largest stride (4), should be fine.
        x_final = torch.cat([u1, u2, u3], dim=1)
        return x_final


class SSDHead(nn.Module):
    def __init__(self):
        super().__init__()

        input_channels = sum(NUM_UPSAMPLE_FILTERS)

        # Calculate total anchors per location
        # Each class has a config.
        # We assume we output one big tensor and slice it, or we use grouped heads.
        # For simplicity, we use one head predicting all classes.
        # But classes have different anchor counts?
        # In this config, all classes have 1 size list (len 1) and 2 rotations.
        # So 2 anchors per class.
        # Total anchors = Num_Classes * 2

        self.num_classes = len(CLASS_NAMES)
        self.num_anchors_per_cls = 2  # 1 size * 2 rotations
        self.total_anchors = self.num_classes * self.num_anchors_per_cls

        # Box Regression: 7 per anchor
        self.conv_box = nn.Conv2d(input_channels, self.total_anchors * 7, 1)

        # Classification: Num_Classes per anchor?
        # Standard SSD: Each anchor is assigned a specific class type in configuration.
        # So we predict 1 score per anchor (is it this class or background?).
        # So output channels = Total_Anchors * 1 (binary) or Total_Anchors * Num_Classes?
        # In PointPillars, usually anchors are class-specific.
        # So for the "Car" anchor, we predict P(Car).
        # So output is Total_Anchors.
        self.conv_cls = nn.Conv2d(input_channels, self.total_anchors, 1)

        # Direction Classification: 2 bins per anchor
        self.conv_dir = nn.Conv2d(input_channels, self.total_anchors * 2, 1)

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)

    def forward(self, x):
        box_preds = self.conv_box(x)
        cls_preds = self.conv_cls(x)
        dir_preds = self.conv_dir(x)

        # Permute to (B, H, W, C) for easier reshaping
        B, _, H, W = box_preds.shape

        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        dir_preds = dir_preds.permute(0, 2, 3, 1).contiguous()

        # Reshape to (B, H*W*Num_Anchors, Code_Size)
        box_preds = box_preds.view(B, -1, 7)
        cls_preds = cls_preds.view(
            B, -1, 1
        )  # Score for the specific class of that anchor
        dir_preds = dir_preds.view(B, -1, 2)

        return cls_preds, box_preds, dir_preds


class PointPillars(nn.Module):
    def __init__(self):
        super().__init__()
        self.pfn = PillarFeatureNet()
        self.scatter = PointPillarsScatter()
        self.backbone = Backbone()
        self.head = SSDHead()

        # Generate Anchors
        self.anchors = self._generate_anchors()  # (H*W*Num_Anchors, 7)
        self.anchors = self.anchors.cuda()  # Move to GPU later or now if available

    def _generate_anchors(self):
        # Grid dimensions
        nx, ny = GRID_SIZE[0], GRID_SIZE[1]

        # Meshgrid
        x = np.linspace(
            POINT_CLOUD_RANGE[0] + VOXEL_SIZE[0] / 2,
            POINT_CLOUD_RANGE[3] - VOXEL_SIZE[0] / 2,
            nx,
        )
        y = np.linspace(
            POINT_CLOUD_RANGE[1] + VOXEL_SIZE[1] / 2,
            POINT_CLOUD_RANGE[4] - VOXEL_SIZE[1] / 2,
            ny,
        )

        # (ny, nx)
        xv, yv = np.meshgrid(x, y)

        anchors_list = []

        # Order must match the head output order
        # Head output: (B, H, W, Total_Anchors)
        # Total_Anchors = Class1_Rot1, Class1_Rot2, Class2_Rot1, ...

        for cfg in ANCHOR_CONFIGS:
            z = cfg["anchor_bottom_heights"][0]  # Assume 1 height
            w, l, h = cfg["anchor_sizes"][0]  # Assume 1 size
            rots = cfg["anchor_rotations"]

            for r in rots:
                # Create anchor grid for this config
                # Shape (ny, nx, 7)

                # x, y, z, w, l, h, yaw
                anchor_grid = np.zeros((ny, nx, 7), dtype=np.float32)
                anchor_grid[:, :, 0] = xv
                anchor_grid[:, :, 1] = yv
                anchor_grid[:, :, 2] = z + h / 2  # Center Z
                anchor_grid[:, :, 3] = w
                anchor_grid[:, :, 4] = l
                anchor_grid[:, :, 5] = h
                anchor_grid[:, :, 6] = r

                anchors_list.append(anchor_grid.reshape(-1, 7))

        # Stack: (Num_Types, H*W, 7) -> (H*W, Num_Types, 7) -> Flatten
        # We need to match the head's flatten order: (H, W, C) -> view(-1, C)
        # So we should stack along the last dim first?
        # Head: (B, H, W, A*7)
        # So at each pixel (y,x), we have A anchors.

        # Current anchors_list is [ (H*W, 7), (H*W, 7), ... ]
        # We want to interleave them pixel by pixel.
        # Stack to (Num_Types, H*W, 7)
        anchors_stacked = np.stack(anchors_list, axis=0)
        # Transpose to (H*W, Num_Types, 7)
        anchors_stacked = anchors_stacked.transpose(1, 0, 2)
        # Flatten
        anchors = anchors_stacked.reshape(-1, 7)

        return torch.from_numpy(anchors).float()

    def forward(self, batch_dict):
        pillar_features = batch_dict["pillar_features"]
        pillar_coords = batch_dict["pillar_coords"]
        num_points = batch_dict["num_points"]
        batch_size = batch_dict["batch_size"]

        # 1. PFN
        x = self.pfn(pillar_features, num_points, pillar_coords)

        # 2. Scatter
        x = self.scatter(x, pillar_coords, batch_size)

        # 3. Backbone
        x = self.backbone(x)

        # 4. Head
        cls_preds, box_preds, dir_preds = self.head(x)

        # Ensure anchors are on same device
        if self.anchors.device != x.device:
            self.anchors = self.anchors.to(x.device)

        if self.training:
            return self.get_loss(cls_preds, box_preds, dir_preds, batch_dict)
        else:
            return self.get_predictions(cls_preds, box_preds, dir_preds, batch_dict)

    def get_loss(self, cls_preds, box_preds, dir_preds, batch_dict):
        gt_boxes = batch_dict["gt_boxes"]  # List of tensors
        gt_labels = batch_dict["gt_labels"]
        batch_size = len(gt_boxes)

        total_cls_loss = 0
        total_box_loss = 0
        total_dir_loss = 0

        # Process each sample in batch (simpler than vectorized matching for variable GT)
        for b in range(batch_size):
            # Get predictions for this sample
            b_cls_preds = cls_preds[b]  # (N_Anchors, 1)
            b_box_preds = box_preds[b]  # (N_Anchors, 7)
            b_dir_preds = dir_preds[b]  # (N_Anchors, 2)

            b_gt_boxes = gt_boxes[b].to(cls_preds.device)
            b_gt_labels = gt_labels[b].to(cls_preds.device)

            if len(b_gt_boxes) == 0:
                # No objects, all background
                # Focal loss with target 0
                # Sigmoid focal loss
                probs = torch.sigmoid(b_cls_preds)
                alpha = 0.25
                gamma = 2.0
                # Target is 0
                loss = -(1 - alpha) * (probs**gamma) * torch.log(1 - probs + 1e-6)
                total_cls_loss += loss.mean()
                continue

            # Match Anchors
            # We need to associate each anchor with a class type to know which GT it can match
            # self.anchors has shape (H*W*Num_Types, 7)
            # We can determine the class index of each anchor
            # Num_Types = len(ANCHOR_CONFIGS) * 2

            # Optimization: Only compute IoU for anchors near GT
            # Calculate distance
            # anchors_xy = self.anchors[:, :2]
            # gt_xy = b_gt_boxes[:, :2]
            # dists = torch.cdist(anchors_xy, gt_xy) # (N_Anchors, N_GT)
            # mask = dists < 4.0 # Filter far away

            # Since 7M anchors is too many for cdist, we assume sparse matching or just random sampling for background?
            # Actually, standard PointPillars implementation uses full matching but on downsampled grid.
            # Here grid is 640x640.
            # We will use a simplified matching:
            # 1. Assign GT to nearest anchors (spatial)
            # 2. Check IoU

            # Due to runtime constraints, we select a subset of anchors to calculate loss on
            # Or we assume the network is fully convolutional and we can just use the dense output.
            # But we need targets.

            # Let's use a simplified strategy:
            # Only consider anchors that have centers close to GT centers.

            # Find indices of anchors close to any GT
            # Grid coords of GT
            gt_x_idx = (
                (b_gt_boxes[:, 0] - POINT_CLOUD_RANGE[0]) / VOXEL_SIZE[0]
            ).long()
            gt_y_idx = (
                (b_gt_boxes[:, 1] - POINT_CLOUD_RANGE[1]) / VOXEL_SIZE[1]
            ).long()

            # Clamp
            gt_x_idx = torch.clamp(gt_x_idx, 0, GRID_SIZE[0] - 1)
            gt_y_idx = torch.clamp(gt_y_idx, 0, GRID_SIZE[1] - 1)

            # Gather anchors around these locations (e.g. 3x3 window)
            # This is an approximation to speed up training
            pos_candidates = []
            for i in range(len(b_gt_boxes)):
                x, y = gt_x_idx[i], gt_y_idx[i]
                # Get linear indices for this pixel's anchors
                # Pixel index = y * nx + x
                # Anchor indices start at pixel_idx * num_types
                pixel_idx = y * GRID_SIZE[0] + x
                num_types = len(ANCHOR_CONFIGS) * 2
                start = pixel_idx * num_types
                end = start + num_types
                pos_candidates.append(torch.arange(start, end, device=cls_preds.device))

            if len(pos_candidates) > 0:
                candidate_indices = torch.cat(pos_candidates)
                candidate_anchors = self.anchors[candidate_indices]

                # IoU
                # Expand dims for broadcasting
                # anchors: (M, 7), gt: (N, 7)
                # We iterate GTs to save memory

                # Targets
                cls_targets = torch.zeros_like(b_cls_preds)  # Default 0 (background)
                reg_targets = torch.zeros_like(b_box_preds)
                dir_targets = torch.zeros(
                    (len(b_cls_preds),), dtype=torch.long, device=cls_preds.device
                )

                # Mask for care/don't care
                # -1: ignore, 0: bg, 1: fg
                cls_weights = torch.zeros((len(b_cls_preds),), device=cls_preds.device)

                # Default background weight
                # We sub-sample background to handle imbalance if needed, or use Focal Loss
                # Focal loss handles all background.

                # Assign Matches
                # For each GT, find best matching anchor of correct class
                # Also assign any anchor with IoU > threshold

                # Map anchor index to class ID
                # 0,1 -> Class 0; 2,3 -> Class 1 ...
                anchor_cls_ids = (
                    torch.arange(num_types, device=cls_preds.device) // 2
                ).repeat(GRID_SIZE[0] * GRID_SIZE[1])
                # Note: This reconstruction is expensive.
                # Better: candidate_indices % num_types // 2

                cand_cls_ids = (candidate_indices % num_types) // 2

                for i in range(len(b_gt_boxes)):
                    gt = b_gt_boxes[i]
                    gt_cls = b_gt_labels[i] - 1  # 0-indexed

                    # Filter candidates by class
                    cls_mask = cand_cls_ids == gt_cls
                    if not cls_mask.any():
                        continue

                    relevant_indices = candidate_indices[cls_mask]
                    relevant_anchors = self.anchors[relevant_indices]

                    # Compute IoU (BEV)
                    # Simplified BEV IoU for speed
                    # Intersection
                    # max(min) - min(max)

                    # Vectorized BEV IoU
                    # Anchors: [x, y, z, w, l, h, r]
                    # GT: [x, y, z, w, l, h, r]
                    # Assume aligned for fast check? No, rotation matters.
                    # Use utils function? It's pair-wise.
                    # Use approximation: Axis Aligned IoU

                    # We will use the provided box_iou_3d_pair in a loop for the small set of candidates
                    ious = []
                    for ra in relevant_anchors:
                        ious.append(box_iou_3d_pair(ra.cpu().numpy(), gt.cpu().numpy()))
                    ious = torch.tensor(ious, device=cls_preds.device)

                    # Thresholds
                    cfg = ANCHOR_CONFIGS[gt_cls]
                    pos_thresh = cfg["matched_threshold"]
                    neg_thresh = cfg["unmatched_threshold"]

                    # Assign Positive
                    pos_mask = ious > pos_thresh

                    # Also assign best match if no pos
                    if not pos_mask.any():
                        best_idx = torch.argmax(ious)
                        pos_mask[best_idx] = True

                    match_indices = relevant_indices[pos_mask]

                    # Set Targets
                    cls_targets[match_indices] = 1.0

                    # Regression Targets
                    encoded_boxes = box_encode(
                        gt.unsqueeze(0).repeat(len(match_indices), 1),
                        self.anchors[match_indices],
                    )
                    reg_targets[match_indices] = encoded_boxes

                    # Direction Targets
                    # 1 if yaw > 0 relative to anchor?
                    # Use simple binning: 0 if yaw in [-pi/2, pi/2], 1 otherwise?
                    # Or relative to anchor orientation.
                    # PointPillars: sin(yaw - anchor_yaw) > 0
                    anchor_yaws = self.anchors[match_indices, 6]
                    dir_mask = torch.sin(gt[6] - anchor_yaws) > 0
                    dir_targets[match_indices] = dir_mask.long()

                    # Weights: 1.0 for Pos
                    cls_weights[match_indices] = 1.0

                    # Negatives: Handled by Focal Loss on all anchors
                    # But we might want to ignore ambiguous ones (between thresholds)
                    ignore_mask = (ious > neg_thresh) & (ious < pos_thresh)
                    ignore_indices = relevant_indices[ignore_mask]
                    # We can set weight to 0 for these?
                    # But Focal Loss is usually applied to all.
                    # Standard practice: ignore region gets 0 weight in loss.
                    # We need a mask for loss.

                # Compute Losses

                # 1. Classification (Focal Loss)
                # p_t = p if y=1 else 1-p
                probs = torch.sigmoid(b_cls_preds).squeeze()
                targets = cls_targets.squeeze()

                alpha = 0.25
                gamma = 2.0

                pt = torch.where(targets == 1, probs, 1 - probs)
                focal_weight = (1 - pt) ** gamma

                # Alpha weighting
                alpha_weight = torch.where(targets == 1, alpha, 1 - alpha)

                # BCE
                bce_loss = -torch.log(pt + 1e-6)

                cls_loss = focal_weight * alpha_weight * bce_loss

                # Normalize by num positive matches
                num_pos = (targets == 1).sum()
                if num_pos > 0:
                    total_cls_loss += cls_loss.sum() / num_pos
                else:
                    total_cls_loss += cls_loss.sum()  # Should be small/zero if perfect

                # 2. Regression (Smooth L1)
                # Only on positives
                pos_mask = targets == 1
                if pos_mask.any():
                    reg_pred = b_box_preds[pos_mask]
                    reg_target = reg_targets[pos_mask]

                    reg_loss = F.smooth_l1_loss(reg_pred, reg_target, reduction="mean")
                    total_box_loss += reg_loss

                    # 3. Direction
                    dir_pred = b_dir_preds[pos_mask]
                    dir_target = dir_targets[pos_mask]

                    dir_loss = F.cross_entropy(dir_pred, dir_target, reduction="mean")
                    total_dir_loss += dir_loss

            else:
                # No candidates found (rare if GT exists)
                pass

        # Average over batch
        loss_dict = {
            "cls_loss": total_cls_loss / batch_size * LOSS_WEIGHTS["cls_weight"],
            "box_loss": total_box_loss / batch_size * LOSS_WEIGHTS["box_weight"],
            "dir_loss": total_dir_loss / batch_size * LOSS_WEIGHTS["dir_weight"],
        }
        loss_dict["loss"] = (
            loss_dict["cls_loss"] + loss_dict["box_loss"] + loss_dict["dir_loss"]
        )

        return loss_dict

    def get_predictions(self, cls_preds, box_preds, dir_preds, batch_dict):
        batch_size = len(cls_preds)
        results = []

        for b in range(batch_size):
            scores = torch.sigmoid(cls_preds[b]).squeeze()

            # Filter low scores
            mask = scores > SCORE_THRESHOLD
            if not mask.any():
                results.append(None)
                continue

            scores = scores[mask]
            reg_preds = box_preds[b][mask]
            dir_preds_b = dir_preds[b][mask]
            anchors_b = self.anchors[mask]

            # Decode boxes
            boxes = box_decode(reg_preds, anchors_b)

            # Resolve direction
            # dir_preds is (N, 2) logits
            dir_labels = torch.argmax(dir_preds_b, dim=1)
            # If label is 1, orientation is opposite?
            # Usually we flip yaw if dir_label is 1 (or 0 depending on training)
            # Training: sin(yaw - anchor) > 0 -> 1.
            # So if 1, we want yaw such that sin > 0.
            # The decoded yaw is reg + anchor.
            # If pred is 1 but sin(decoded - anchor) < 0, we flip?
            # Simple approach: If dir_labels == 1 and current orientation is 'negative', flip.
            # Easier: Just rely on regression. Direction classifier is auxiliary.
            # But we can use it to correct 180 flips.

            # NMS
            # Convert to numpy for utils
            boxes_np = boxes.detach().cpu().numpy()
            scores_np = scores.detach().cpu().numpy()

            keep = nms_3d(
                boxes_np,
                scores_np,
                iou_threshold=NMS_IOU_THRESHOLD,
                max_dets=MAX_DETECTIONS,
            )

            if len(keep) > 0:
                final_boxes = boxes_np[keep]
                final_scores = scores_np[keep]

                # Determine class names
                # We need to map back from anchor index to class
                # Indices in original anchor tensor
                full_indices = torch.nonzero(mask, as_tuple=False).squeeze()
                if full_indices.dim() == 0:
                    full_indices = full_indices.unsqueeze(0)
                kept_full_indices = full_indices[keep]

                num_types = len(ANCHOR_CONFIGS) * 2
                # anchor_idx % num_types gives the type index
                # type_idx // 2 gives the class index
                type_indices = kept_full_indices % num_types
                class_indices = type_indices // 2

                class_names = [CLASS_NAMES[i] for i in class_indices.cpu().numpy()]

                pred_string = ""
                for i in range(len(final_boxes)):
                    # conf x y z w l h yaw class
                    box = final_boxes[i]
                    pred_string += f"{final_scores[i]:.4f} {box[0]:.4f} {box[1]:.4f} {box[2]:.4f} {box[3]:.4f} {box[4]:.4f} {box[5]:.4f} {box[6]:.4f} {class_names[i]} "

                results.append(pred_string.strip())
            else:
                results.append("")

        return results
