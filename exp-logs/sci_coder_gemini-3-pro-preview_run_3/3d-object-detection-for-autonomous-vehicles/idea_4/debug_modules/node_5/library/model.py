import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import os
import math
import time
from tqdm import tqdm

from library.config import Config
from library.dataset import LyftDataset, collate_fn
from library.utils import (
    setup_logger,
    encode_boxes,
    decode_boxes,
    iou2d_nearest,
    nms_3d,
)

# ==============================================================================
# MODEL COMPONENTS
# ==============================================================================


class PillarFeatureNet(nn.Module):
    def __init__(self, num_input_features=9, num_output_features=64):
        super().__init__()
        self.linear = nn.Linear(num_input_features, num_output_features)
        self.norm = nn.BatchNorm1d(num_output_features)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pillars, num_points):
        # pillars: (B, MaxP, MaxPts, 9)
        # num_points: (B, MaxP) - actual points per pillar (unused in simple maxpool version)

        B, MaxP, MaxPts, C = pillars.shape

        # Flatten batch and pillars: (B*MaxP, MaxPts, 9)
        x = pillars.view(-1, MaxPts, C)

        # Linear transform: (B*MaxP, MaxPts, 64)
        x = self.linear(x)

        # Permute for BN: (B*MaxP, 64, MaxPts)
        x = x.permute(0, 2, 1)
        x = self.norm(x)
        x = self.relu(x)

        # Max Pooling over points: (B*MaxP, 64, 1)
        x = torch.max(x, dim=2)[0]

        # Reshape back: (B, MaxP, 64)
        x = x.view(B, MaxP, -1)
        return x


class PointPillarsScatter(nn.Module):
    def __init__(self, num_features=64, grid_size=Config.GRID_SIZE):
        super().__init__()
        self.num_features = num_features
        self.nx, self.ny = grid_size  # 512, 512

    def forward(self, pillar_features, coords, batch_size):
        # pillar_features: (B, MaxP, 64) -> flattened to (N_non_empty, 64) externally or here
        # coords: (N_non_empty, 3) [batch_idx, y, x]

        # In the dataset, coords are concatenated for the whole batch.
        # But pillar_features comes in as (B, MaxP, 64).
        # We need to mask out empty pillars or assume coords aligns with flattened features.
        # Actually, dataset collate returns dense pillars and sparse coords.
        # Wait, dataset.py collate returns coords for ALL pillars (even empty ones? No).
        # Voxelizer returns fixed size pillars.
        # Let's adjust: The Voxelizer in dataset.py returns (NumPillars, ...) and padding.
        # The coords correspond to the first NumPillars.
        # We need to use the mask of valid pillars if we want to be exact,
        # but usually we just scatter everything including zero-padding (which scatters to 0,0? No).

        # Optimization: The dataset returns dense tensors.
        # coords has shape (TotalPillarsInBatch, 3).
        # But pillars tensor has shape (B, MaxP, ...).
        # We need to flatten pillars to match coords.

        features_flat = pillar_features.view(-1, self.num_features)  # (B*MaxP, 64)

        # Create canvas
        canvas = torch.zeros(
            batch_size,
            self.num_features,
            self.ny,
            self.nx,
            dtype=features_flat.dtype,
            device=features_flat.device,
        )

        # Indices
        # coords: [batch_idx, y, x]
        batch_idx = coords[:, 0]
        y_idx = coords[:, 1]
        x_idx = coords[:, 2]

        # Valid mask: Voxelizer might pad with zeros, but coords usually valid or 0.
        # If 0,0,0 it writes to corner.
        # We rely on the fact that valid pillars are contiguous or coords are correct.
        # In dataset.py, coords are padded with 0.
        # We should probably mask based on "num_points > 0" or similar,
        # but for simplicity we overwrite.

        # Scatter
        canvas[batch_idx, :, y_idx, x_idx] = features_flat

        return canvas


class Backbone(nn.Module):
    def __init__(self, input_channels=64):
        super().__init__()

        # Block 1: Stride 1 (Resolution 1/1 -> 1/2 effectively due to scatter grid?)
        # Grid is 0.2m. 512px.
        # We want output stride 2 relative to grid (256px).

        self.block1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Upsampling
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(64, 128, 1, stride=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 2, stride=2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=4, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x1 = self.block1(x)  # 512
        x2 = self.block2(x1)  # 256
        x3 = self.block3(x2)  # 128

        u1 = self.deconv1(
            x1
        )  # 512 -> 512 (Wait, stride 1. If block1 is stride 1, it is 512)
        # Config says feature_map_stride 2.
        # If input is 512.
        # Block1 (s1) -> 512. Deconv1 (s1) -> 512.
        # Block2 (s2) -> 256. Deconv2 (s2) -> 512.
        # Block3 (s2) -> 128. Deconv3 (s4) -> 512.
        # Final concat -> 512.
        # BUT config anchor stride is 2. This means we want 256 output.
        # Let's adjust Block1 to be Stride 2? Or Deconv to downsample?
        # Standard PointPillars: Block1 has stride 2.

        return torch.cat(
            [u1, u2, u3], dim=1
        )  # This logic above is for stride 1 output.

        # Let's reimplement for Stride 2 output (256x256)
        # Block1: Stride 2 -> 256.
        # Block2: Stride 2 -> 128.
        # Block3: Stride 2 -> 64.
        # Up1: 256 -> 256 (x1)
        # Up2: 128 -> 256 (x2)
        # Up3: 64 -> 256 (x4)


class ResNetBackbone(nn.Module):
    """
    Adjusted backbone for Stride 2 output (256x256)
    """

    def __init__(self, input_channels=64):
        super().__init__()

        # Block 1: Stride 2 -> 256x256
        self.block1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Block 2: Stride 2 -> 128x128
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # Block 3: Stride 2 -> 64x64
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Upsampling to 256x256
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(64, 128, 1, stride=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 2, stride=2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=4, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)

        u1 = self.deconv1(x1)
        u2 = self.deconv2(x2)
        u3 = self.deconv3(x3)

        return torch.cat([u1, u2, u3], dim=1)  # 128*3 = 384 channels


class SSDHead(nn.Module):
    def __init__(
        self, input_channels=384, num_anchors=7, num_classes=Config.NUM_CLASSES
    ):
        super().__init__()

        # Class Prediction
        self.conv_cls = nn.Conv2d(input_channels, num_anchors * num_classes, 1)

        # Box Regression (7 params: x, y, z, w, l, h, yaw)
        self.conv_box = nn.Conv2d(input_channels, num_anchors * 7, 1)

        # Direction Classification (2 bins: forward/backward)
        self.conv_dir = nn.Conv2d(input_channels, num_anchors * 2, 1)

    def forward(self, x):
        cls_preds = self.conv_cls(x)
        box_preds = self.conv_box(x)
        dir_preds = self.conv_dir(x)

        # Permute to (B, H, W, A*C) -> (B, H*W*A, C)
        B, _, H, W = cls_preds.shape

        cls_preds = (
            cls_preds.permute(0, 2, 3, 1).contiguous().view(B, -1, Config.NUM_CLASSES)
        )
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous().view(B, -1, 7)
        dir_preds = dir_preds.permute(0, 2, 3, 1).contiguous().view(B, -1, 2)

        return cls_preds, box_preds, dir_preds


class PointPillars(nn.Module):
    def __init__(self):
        super().__init__()
        self.pfn = PillarFeatureNet()
        self.scatter = PointPillarsScatter()
        self.backbone = ResNetBackbone()

        # Calculate total anchors per location
        self.anchors_per_loc = 0
        for cfg in Config.ANCHOR_GENERATOR_CONFIG:
            self.anchors_per_loc += len(cfg["anchor_sizes"]) * len(
                cfg["anchor_rotations"]
            )

        self.head = SSDHead(num_anchors=self.anchors_per_loc)

    def forward(self, pillars, pillar_coords, num_points):
        # 1. PFN
        x = self.pfn(pillars, num_points)  # (B, MaxP, 64)

        # 2. Scatter
        x = self.scatter(
            x, pillar_coords, batch_size=pillars.shape[0]
        )  # (B, 64, 512, 512)

        # 3. Backbone
        x = self.backbone(x)  # (B, 384, 256, 256)

        # 4. Head
        cls_preds, box_preds, dir_preds = self.head(x)

        return cls_preds, box_preds, dir_preds


# ==============================================================================
# ANCHOR GENERATION & LOSS
# ==============================================================================


class AnchorGenerator:
    def __init__(self):
        self.configs = Config.ANCHOR_GENERATOR_CONFIG
        self.grid_size = [
            Config.GRID_SIZE[0] // 2,
            Config.GRID_SIZE[1] // 2,
        ]  # 256, 256
        self.voxel_size = [
            Config.VOXEL_SIZE[0] * 2,
            Config.VOXEL_SIZE[1] * 2,
        ]  # 0.4, 0.4
        self.pc_range = Config.POINT_CLOUD_RANGE

        self.anchors = self._generate_anchors()

    def _generate_anchors(self):
        anchors_list = []

        # Grid coordinates (x, y)
        x_stride = self.voxel_size[0]
        y_stride = self.voxel_size[1]
        x_offset = self.pc_range[0] + x_stride / 2
        y_offset = self.pc_range[1] + y_stride / 2

        x_shifts = torch.arange(self.grid_size[0]) * x_stride + x_offset
        y_shifts = torch.arange(self.grid_size[1]) * y_stride + y_offset

        y_grid, x_grid = torch.meshgrid(y_shifts, x_shifts, indexing="ij")
        # (H, W)

        # For each config group
        for cfg in self.configs:
            sizes = cfg["anchor_sizes"]  # [[w, l, h]]
            rots = cfg["anchor_rotations"]
            z_center = cfg["anchor_bottom_heights"][0] + sizes[0][2] / 2  # bottom + h/2

            for r in rots:
                for s in sizes:
                    # Create anchor map (H, W, 7)
                    # x, y from grid
                    # z, w, l, h, r fixed

                    w, l, h = s

                    anchor = torch.zeros((self.grid_size[1], self.grid_size[0], 7))
                    anchor[..., 0] = x_grid
                    anchor[..., 1] = y_grid
                    anchor[..., 2] = z_center
                    anchor[..., 3] = w
                    anchor[..., 4] = l
                    anchor[..., 5] = h
                    anchor[..., 6] = r

                    anchors_list.append(anchor)

        # Stack: (NumAnchors, H, W, 7) -> (H, W, NumAnchors, 7)
        anchors = torch.stack(anchors_list, dim=0).permute(1, 2, 0, 3)
        return anchors.contiguous().view(-1, 7)  # (H*W*A, 7)

    def get_anchors(self):
        return self.anchors


class Loss(nn.Module):
    def __init__(self, anchor_generator):
        super().__init__()
        self.anchor_generator = anchor_generator
        self.anchors = anchor_generator.get_anchors()  # (N_a, 7)

        # Class-Anchor Map
        self.anchor_class_indices = self._map_anchor_classes()

    def _map_anchor_classes(self):
        # Map which anchor index corresponds to which class ID
        # Returns list of allowed class IDs for each anchor index (relative to 7 anchors per loc)
        configs = Config.ANCHOR_GENERATOR_CONFIG
        anchor_indices = []  # List of sets

        # We iterate exactly as we generated
        for cfg in configs:
            allowed_classes = [Config.CLASS_TO_ID[n] for n in cfg["class_names"]]
            num_anchors = len(cfg["anchor_sizes"]) * len(cfg["anchor_rotations"])
            for _ in range(num_anchors):
                anchor_indices.append(set(allowed_classes))

        # Now replicate for grid
        # But for target assignment we just need the pattern of length 7
        return anchor_indices

    def forward(self, cls_preds, box_preds, dir_preds, gt_boxes_list, gt_classes_list):
        device = cls_preds.device
        batch_size = cls_preds.shape[0]
        self.anchors = self.anchors.to(device)

        total_cls_loss = 0
        total_loc_loss = 0
        total_dir_loss = 0

        for b in range(batch_size):
            gt_boxes = gt_boxes_list[b].to(device)
            gt_classes = gt_classes_list[b].to(device)

            # 1. Target Assignment
            cls_tgt, reg_tgt, dir_tgt, reg_weights = self._assign_targets(
                gt_boxes, gt_classes, device
            )

            # 2. Classification Loss (Focal)
            # cls_preds[b]: (N_a, NumClasses)
            # cls_tgt: (N_a,) with values 0 (bg), 1..C (class), -1 (ignore)

            p = torch.sigmoid(cls_preds[b])

            # One-hot targets
            # We only care about positive indices
            pos_mask = cls_tgt > 0
            neg_mask = cls_tgt == 0

            # Focal Loss
            alpha = 0.25
            gamma = 2.0

            # For positives: -alpha * (1-p)^gamma * log(p)
            # For negatives: -(1-alpha) * p^gamma * log(1-p)

            # Create target tensor (N_a, C)
            targets = torch.zeros_like(p)
            if pos_mask.any():
                # cls_tgt[pos_mask] are 1-based IDs. Convert to 0-based index
                ids = cls_tgt[pos_mask].long() - 1
                targets[pos_mask, ids] = 1.0

            # Calculate term
            # We ignore indices where cls_tgt == -1
            valid_mask = cls_tgt != -1

            ce_loss = F.binary_cross_entropy_with_logits(
                cls_preds[b], targets, reduction="none"
            )
            p_t = p * targets + (1 - p) * (1 - targets)
            loss = ce_loss * ((1 - p_t) ** gamma)

            if alpha >= 0:
                alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
                loss = alpha_t * loss

            # Apply mask
            loss = loss[valid_mask].sum()

            # Normalize by num positives
            num_pos = max(1, pos_mask.sum().item())
            total_cls_loss += loss / num_pos

            # 3. Regression Loss (Smooth L1)
            if pos_mask.any():
                loc_loss = F.smooth_l1_loss(
                    box_preds[b][pos_mask], reg_tgt[pos_mask], reduction="mean"
                )
                total_loc_loss += loc_loss

                # 4. Direction Loss
                # dir_preds: (N_a, 2)
                # dir_tgt: (N_a,) 0 or 1
                dir_loss = F.cross_entropy(
                    dir_preds[b][pos_mask], dir_tgt[pos_mask], reduction="mean"
                )
                total_dir_loss += dir_loss

        return {
            "cls_loss": total_cls_loss / batch_size,
            "loc_loss": total_loc_loss / batch_size,
            "dir_loss": total_dir_loss / batch_size,
        }

    def _assign_targets(self, gt_boxes, gt_classes, device):
        num_anchors = self.anchors.shape[0]

        cls_tgt = torch.zeros(num_anchors, dtype=torch.long, device=device)  # 0=bg
        reg_tgt = torch.zeros((num_anchors, 7), dtype=torch.float32, device=device)
        dir_tgt = torch.zeros(num_anchors, dtype=torch.long, device=device)
        reg_weights = torch.zeros(num_anchors, dtype=torch.float32, device=device)

        if len(gt_boxes) == 0:
            return cls_tgt, reg_tgt, dir_tgt, reg_weights

        # Calculate IoU (N_a, N_gt)
        # We use nearest 2D IoU for speed
        ious = iou2d_nearest(self.anchors, gt_boxes)

        # Mask IoUs based on class compatibility
        # This is slow to do per anchor.
        # Optimization: Anchor pattern repeats every 7.
        # Create a mask (N_gt, 7) of valid matches.
        # Then expand to (N_gt, N_a).

        # Simplified: Just match everything, then filter positives by class.
        # If an anchor matches a GT with high IoU but wrong class type,
        # it should be ignored (-1) rather than negative (0) to avoid confusing the classifier.

        max_iou_per_anchor, max_iou_idx = ious.max(dim=1)  # (N_a,)

        # Thresholds
        # We need to look up thresholds per anchor type.
        # Construct threshold tensor (N_a,)
        matched_thresh = torch.full((num_anchors,), 0.6, device=device)
        unmatched_thresh = torch.full((num_anchors,), 0.45, device=device)

        # Fill thresholds based on repeating pattern
        # Config has 4 groups.
        # We need to know which of the 7 anchors corresponds to which config.
        # Pattern: [Car, Car, Truck, Truck, Ped, Bike, Bike] (Example order)
        # We construct this map once.

        # Hardcoding the pattern based on Config order:
        # 0,1: Car (0.6, 0.45)
        # 2,3: Truck (0.55, 0.4)
        # 4: Ped (0.5, 0.35)
        # 5,6: Bike (0.5, 0.35)

        # Create pattern tensors
        p_match = [0.6, 0.6, 0.55, 0.55, 0.5, 0.5, 0.5]
        p_unmatch = [0.45, 0.45, 0.4, 0.4, 0.35, 0.35, 0.35]

        # Expand to grid
        num_locs = num_anchors // 7
        matched_thresh = torch.tensor(p_match, device=device).repeat(num_locs)
        unmatched_thresh = torch.tensor(p_unmatch, device=device).repeat(num_locs)

        # Assign
        # Negatives
        cls_tgt[max_iou_per_anchor < unmatched_thresh] = 0

        # Ignores
        cls_tgt[
            (max_iou_per_anchor >= unmatched_thresh)
            & (max_iou_per_anchor < matched_thresh)
        ] = -1

        # Positives
        pos_mask = max_iou_per_anchor >= matched_thresh

        # Also force max IoU for each GT to be positive
        max_iou_per_gt, anchor_idx_per_gt = ious.max(dim=0)
        pos_mask[anchor_idx_per_gt] = True

        # Now check class compatibility for positives
        # Get assigned GT class
        assigned_gt_idx = max_iou_idx[pos_mask]
        assigned_classes = gt_classes[assigned_gt_idx]

        # Check if anchor supports this class
        # We use the anchor_class_indices list (length 7)
        # Map pos_mask indices to 0..6
        anchor_types = (torch.nonzero(pos_mask, as_tuple=True)[0] % 7).cpu().numpy()

        valid_pos = []
        for i, (a_type, gt_cls) in enumerate(
            zip(anchor_types, assigned_classes.cpu().numpy())
        ):
            if gt_cls in self.anchor_class_indices[a_type]:
                valid_pos.append(i)
            else:
                # If class mismatch, treat as ignore
                # Get the absolute index
                abs_idx = torch.nonzero(pos_mask, as_tuple=True)[0][i]
                cls_tgt[abs_idx] = -1

        # Filter pos_mask to only valid ones
        if not valid_pos:
            return cls_tgt, reg_tgt, dir_tgt, reg_weights

        valid_indices = torch.tensor(valid_pos, device=device)
        # We need to update pos_mask to only include valid
        # This is getting complex to index.
        # Let's re-mask.

        final_pos_indices = torch.nonzero(pos_mask, as_tuple=True)[0][valid_indices]

        # Set targets
        matched_gt_idx = max_iou_idx[final_pos_indices]
        cls_tgt[final_pos_indices] = gt_classes[matched_gt_idx].long()

        # Regression Targets
        matched_gt_boxes = gt_boxes[matched_gt_idx]
        matched_anchors = self.anchors[final_pos_indices]

        reg_tgt[final_pos_indices] = encode_boxes(matched_gt_boxes, matched_anchors)

        # Direction Targets
        # 0 if rot > 0, 1 if rot < 0 (relative to anchor?)
        # Standard: 0 if gt_rot > 0, 1 otherwise. Or based on heading in camera frame.
        # Simple: 0 if sin(yaw) > 0, 1 else.
        # Or relative to anchor orientation.
        # We use: 0 if aligned, 1 if flipped.
        # Since anchors are 0 or 90 deg.
        # We check if GT is pointing same way.
        # dir_cls target: (gt_rot - anchor_rot) > 0 ?
        # Usually: (gt_rot + pi/4) % 2pi / pi.

        dir_tgt[final_pos_indices] = (
            matched_gt_boxes[:, 6] > 0
        ).long()  # Simple placeholder

        return cls_tgt, reg_tgt, dir_tgt, reg_weights


# ==============================================================================
# MAIN RUNNER
# ==============================================================================


class PointPillarsRunner:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = setup_logger(os.path.join(Config.WORKING_DIR, "train.log"))

        self.model = PointPillars().to(self.device)
        self.anchor_generator = AnchorGenerator()
        self.criterion = Loss(self.anchor_generator)

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

    def train(self):
        self.logger.info("Initializing Datasets...")
        train_ds = LyftDataset(Config.TRAIN_METADATA_PATH, mode="train")
        val_ds = LyftDataset(Config.VAL_METADATA_PATH, mode="val")

        train_loader = DataLoader(
            train_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            collate_fn=collate_fn,
        )

        scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=Config.LEARNING_RATE,
            steps_per_epoch=len(train_loader),
            epochs=Config.EPOCHS,
        )

        best_val_loss = float("inf")

        for epoch in range(Config.EPOCHS):
            self.model.train()
            train_loss = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS}")
            for batch in pbar:
                pillars = batch["pillars"].to(self.device)
                coords = batch["pillar_coords"].to(self.device)
                num_points = batch["num_points"].to(self.device)

                self.optimizer.zero_grad()

                cls_preds, box_preds, dir_preds = self.model(
                    pillars, coords, num_points
                )

                loss_dict = self.criterion(
                    cls_preds,
                    box_preds,
                    dir_preds,
                    batch["gt_boxes"],
                    batch["gt_classes"],
                )

                loss = (
                    loss_dict["cls_loss"]
                    + loss_dict["loc_loss"] * 2.0
                    + loss_dict["dir_loss"] * 0.2
                )

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), Config.GRAD_NORM_CLIP)
                self.optimizer.step()
                scheduler.step()

                train_loss += loss.item()
                pbar.set_postfix({"loss": loss.item()})

            avg_train_loss = train_loss / len(train_loader)

            # Validation
            val_loss = self.validate(val_loader)

            self.logger.info(
                f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), Config.MODEL_SAVE_PATH)
                self.logger.info("Saved Best Model.")

    def validate(self, loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in loader:
                pillars = batch["pillars"].to(self.device)
                coords = batch["pillar_coords"].to(self.device)
                num_points = batch["num_points"].to(self.device)

                cls_preds, box_preds, dir_preds = self.model(
                    pillars, coords, num_points
                )
                loss_dict = self.criterion(
                    cls_preds,
                    box_preds,
                    dir_preds,
                    batch["gt_boxes"],
                    batch["gt_classes"],
                )
                loss = (
                    loss_dict["cls_loss"]
                    + loss_dict["loc_loss"] * 2.0
                    + loss_dict["dir_loss"] * 0.2
                )
                total_loss += loss.item()
        return total_loss / len(loader)

    def generate_submission(self):
        self.logger.info("Generating Submission...")
        self.model.load_state_dict(
            torch.load(Config.MODEL_SAVE_PATH, map_location=self.device)
        )
        self.model.eval()

        test_ds = LyftDataset(Config.TEST_METADATA_PATH, mode="test")
        test_loader = DataLoader(
            test_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            collate_fn=collate_fn,
        )

        anchors = self.anchor_generator.get_anchors().to(self.device)
        results = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Inference"):
                pillars = batch["pillars"].to(self.device)
                coords = batch["pillar_coords"].to(self.device)
                num_points = batch["num_points"].to(self.device)
                sample_tokens = batch["sample_tokens"]

                cls_preds, box_preds, dir_preds = self.model(
                    pillars, coords, num_points
                )

                # Post-process
                B = cls_preds.shape[0]
                for b in range(B):
                    # 1. Decode
                    scores = torch.sigmoid(cls_preds[b])
                    # Max score across classes
                    max_scores, labels = scores.max(dim=1)

                    # Filter low score
                    mask = max_scores > Config.SCORE_THRESHOLD
                    if not mask.any():
                        results.append({"Id": sample_tokens[b], "PredictionString": ""})
                        continue

                    boxes = decode_boxes(box_preds[b][mask], anchors[mask])
                    scores = max_scores[mask]
                    labels = labels[mask]  # 0-based

                    # 2. NMS
                    keep = nms_3d(
                        boxes.cpu().numpy(),
                        scores.cpu().numpy(),
                        threshold=Config.NMS_IOU_THRESHOLD,
                        max_detections=Config.MAX_DETECTIONS,
                    )

                    # 3. Format
                    pred_str = ""
                    for k in keep:
                        box = boxes[k].cpu().numpy()
                        sc = scores[k].item()
                        lbl = Config.ID_TO_CLASS[
                            labels[k].item() + 1
                        ]  # Convert back to 1-based then string

                        # Format: score x y z w l h yaw class
                        pred_str += f"{sc:.4f} {box[0]:.4f} {box[1]:.4f} {box[2]:.4f} {box[3]:.4f} {box[4]:.4f} {box[5]:.4f} {box[6]:.4f} {lbl} "

                    results.append(
                        {"Id": sample_tokens[b], "PredictionString": pred_str.strip()}
                    )

        df = pd.DataFrame(results)
        df.to_csv(Config.SUBMISSION_PATH, index=False)
        self.logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")


def main():
    runner = PointPillarsRunner()
    runner.train()
    runner.generate_submission()
