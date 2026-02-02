import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from library.config import Config
from library.utils import box_decode, box_encode, iou3d, nms_3d


class PillarVFE(nn.Module):
    """
    Voxel Feature Encoder: Converts raw point clouds in voxels to point-wise features,
    then aggregates them into voxel-wise features.
    """

    def __init__(self):
        super().__init__()
        self.use_norm = True
        self.with_distance = False

        # Input channels: x, y, z, i (4)
        # Augmented channels: x, y, z, i, x-xc, y-yc, z-zc, x-xp, y-yp, z-zp (10)
        # We use a simplified set: x, y, z, i, x-xc, y-yc, z-zc, x-xp, y-yp (9)
        # xc, yc, zc: arithmetic mean of points in voxel
        # xp, yp: geometric center of voxel

        self.in_channels = 9
        self.out_channels = Config.HIDDEN_DIM

        self.linear = nn.Linear(self.in_channels, self.out_channels)
        self.norm = nn.BatchNorm1d(self.out_channels)

    def forward(self, features, num_points, coors):
        """
        Args:
            features: (M, 32, 4) [x, y, z, i]
            num_points: (M,)
            coors: (M, 4) [batch_idx, z, y, x]
        """
        # 1. Calculate arithmetic mean (xc, yc, zc)
        # Mask out zero-padded points
        points_mean = features[:, :, :3].sum(dim=1, keepdim=True) / num_points.type_as(
            features
        ).view(-1, 1, 1).clamp(min=1.0)

        # 2. Calculate geometric center (xp, yp, zp) - derived from voxel coordinates
        # coors: (batch, z, y, x)
        # We need to convert indices back to world coordinates
        # x_p = x_idx * v_x + min_x + v_x/2
        dtype = features.dtype
        device = features.device

        voxel_size = torch.tensor(Config.VOXEL_SIZE, device=device, dtype=dtype)
        pc_range = torch.tensor(Config.POINT_CLOUD_RANGE, device=device, dtype=dtype)

        # coors[:, 3] is x_idx, coors[:, 2] is y_idx, coors[:, 1] is z_idx
        x_idx = coors[:, 3].type_as(features)
        y_idx = coors[:, 2].type_as(features)
        # z_idx = coors[:, 1].type_as(features) # Not used for BEV usually, but consistent

        x_p = x_idx * voxel_size[0] + pc_range[0] + voxel_size[0] / 2
        y_p = y_idx * voxel_size[1] + pc_range[1] + voxel_size[1] / 2
        # z_p is usually not critical for 2D backbone but we can compute it
        # z_p = z_idx * voxel_size[2] + pc_range[2] + voxel_size[2] / 2

        # Expand for broadcasting
        # (M, 1, 1)
        x_p = x_p.unsqueeze(1).unsqueeze(2)
        y_p = y_p.unsqueeze(1).unsqueeze(2)

        # 3. Augment Features
        # f_cluster = points - mean
        f_cluster = features[:, :, :3] - points_mean

        # f_center = points - center (only x, y)
        f_center_x = features[:, :, 0:1] - x_p
        f_center_y = features[:, :, 1:2] - y_p

        # Combine: [x, y, z, i, x-xc, y-yc, z-zc, x-xp, y-yp]
        features_aug = torch.cat([features, f_cluster, f_center_x, f_center_y], dim=-1)

        # 4. Linear Transformation
        # Flatten: (M * 32, 9)
        M, P, C = features_aug.shape
        x = features_aug.view(-1, C)

        x = self.linear(x)
        x = self.norm(x)
        x = F.relu(x)

        # Reshape back: (M, 32, C_out)
        x = x.view(M, P, -1)

        # 5. Max Pooling
        # We need to mask padded points before max pooling?
        # Since we use ReLU, values are >= 0. Padded zeros (if input was 0) might interfere if features are negative before ReLU.
        # But standard PFN max pools directly.
        # To be safe against padding affecting max, we can replace padded spots with -inf, but usually 0 is fine with ReLU.
        voxel_features = torch.max(x, dim=1)[0]

        return voxel_features


class PointPillarsScatter(nn.Module):
    """
    Scatters voxel features into a 2D pseudo-image (BEV).
    """

    def __init__(self):
        super().__init__()
        self.nx = Config.GRID_SIZE[0]
        self.ny = Config.GRID_SIZE[1]
        self.num_channels = Config.HIDDEN_DIM

    def forward(self, voxel_features, coors, batch_size):
        """
        Args:
            voxel_features: (M, C)
            coors: (M, 4) [batch_idx, z, y, x]
            batch_size: int
        """
        # Create dense canvas
        canvas = torch.zeros(
            batch_size,
            self.num_channels,
            self.ny * self.nx,
            dtype=voxel_features.dtype,
            device=voxel_features.device,
        )

        # Calculate indices in the flattened 2D grid
        # idx = batch_idx * (ny * nx) + y_idx * nx + x_idx
        # But we want (Batch, C, H, W).
        # We can scatter to (Batch, H*W, C) then transpose/reshape, or scatter to (Batch, C, H*W)

        batch_idx = coors[:, 0]
        y_idx = coors[:, 2]
        x_idx = coors[:, 3]

        indices = batch_idx * (self.ny * self.nx) + y_idx * self.nx + x_idx
        indices = indices.long()

        # Scatter
        # voxel_features: (M, C)
        # canvas: (B, C, H*W) -> treat as flat (B*C*H*W)? No.
        # We use index_add_ or scatter_
        # Reshape canvas to (B * H * W, C)
        canvas_flat = canvas.view(-1, self.num_channels)

        # Indices need to be expanded for all channels?
        # Easier: canvas[batch, :, y, x] = feature
        # But scatter operations are faster on flat tensors.

        # Valid indices check
        mask = (indices >= 0) & (indices < (batch_size * self.ny * self.nx))
        indices = indices[mask]
        voxel_features = voxel_features[mask]

        if len(indices) > 0:
            canvas_flat.index_add_(0, indices, voxel_features)

        # Reshape to (B, C, H, W)
        batch_canvas = canvas_flat.view(batch_size, self.ny, self.nx, self.num_channels)
        batch_canvas = batch_canvas.permute(0, 3, 1, 2).contiguous()

        return batch_canvas


class Backbone(nn.Module):
    """
    ResNet-like FPN Backbone.
    """

    def __init__(self):
        super().__init__()

        layer_nums = Config.LAYER_NUMS
        layer_strides = Config.LAYER_STRIDES
        num_filters = Config.NUM_FILTERS
        up_strides = Config.UP_STRIDES
        input_channels = Config.HIDDEN_DIM

        # Block 1
        self.block1 = self._make_layer(
            input_channels, num_filters[0], layer_nums[0], layer_strides[0]
        )
        # Block 2
        self.block2 = self._make_layer(
            num_filters[0], num_filters[1], layer_nums[1], layer_strides[1]
        )
        # Block 3
        self.block3 = self._make_layer(
            num_filters[1], num_filters[2], layer_nums[2], layer_strides[2]
        )

        # Deconvolutions (Upsampling)
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(
                num_filters[0],
                num_filters[0],
                up_strides[0],
                stride=up_strides[0],
                bias=False,
            ),
            nn.BatchNorm2d(num_filters[0]),
            nn.ReLU(inplace=True),
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(
                num_filters[1],
                num_filters[1],
                up_strides[1],
                stride=up_strides[1],
                bias=False,
            ),
            nn.BatchNorm2d(num_filters[1]),
            nn.ReLU(inplace=True),
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(
                num_filters[2],
                num_filters[2],
                up_strides[2],
                stride=up_strides[2],
                bias=False,
            ),
            nn.BatchNorm2d(num_filters[2]),
            nn.ReLU(inplace=True),
        )

        self.out_channels = sum(num_filters)

    def _make_layer(self, in_c, out_c, num_layers, stride):
        layers = []
        # First layer handles stride and channel change
        layers.append(nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_c))
        layers.append(nn.ReLU(inplace=True))

        for _ in range(num_layers - 1):
            layers.append(nn.Conv2d(out_c, out_c, 3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, C, H, W)
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)

        u1 = self.deconv1(x1)
        u2 = self.deconv2(x2)
        u3 = self.deconv3(x3)

        # Concatenate
        out = torch.cat([u1, u2, u3], dim=1)
        return out


class CenterHead(nn.Module):
    """
    Stage 1: Anchor-free proposal generator.
    Predicts Heatmap and Box Regression.
    """

    def __init__(self, input_channels):
        super().__init__()

        self.num_classes = Config.NUM_CLASSES

        # Shared conv
        self.shared = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, 3, padding=1, bias=True),
            nn.BatchNorm2d(input_channels),
            nn.ReLU(inplace=True),
        )

        # Heatmap Head
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, 3, padding=1, bias=True),
            nn.BatchNorm2d(input_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_channels, self.num_classes, 1, bias=True),
        )
        # Initialize bias for heatmap to -2.19 (focal loss trick)
        self.heatmap_head[-1].bias.data.fill_(-2.19)

        # Regression Head
        # Targets: [dx, dy, z, log(w), log(l), log(h), sin(yaw), cos(yaw)] -> 8 channels
        self.reg_head = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, 3, padding=1, bias=True),
            nn.BatchNorm2d(input_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_channels, 8, 1, bias=True),
        )

    def forward(self, x):
        feat = self.shared(x)
        heatmap = self.heatmap_head(feat)
        heatmap = torch.sigmoid(heatmap)  # Normalize to [0, 1]
        regression = self.reg_head(feat)
        return heatmap, regression

    def get_proposals(self, heatmap, regression, topk=500):
        """
        Decode predictions into bounding boxes.
        """
        B, C, H, W = heatmap.shape
        device = heatmap.device

        # 1. Find peaks
        heatmap_max = F.max_pool2d(heatmap, kernel_size=3, stride=1, padding=1)
        keep = (heatmap == heatmap_max) & (heatmap > 0.1)
        heatmap = heatmap * keep.float()

        # 2. Top-K
        # Flatten
        heatmap_flat = heatmap.view(B, -1)
        scores, indices = torch.topk(
            heatmap_flat, k=min(topk, heatmap_flat.shape[1]), dim=1
        )

        # Convert indices to x, y, class
        ys = indices // W
        xs = indices % W
        # Usually heatmap is (B, C, H, W), flattening gives index in C*H*W is tricky for multi-class
        # Let's do topk on (B, C*H*W)
        # But we need class index.

        # Correct approach for multi-class topk:
        scores, indices = torch.topk(
            heatmap.view(B, -1), k=min(topk, heatmap.numel() // B), dim=1
        )

        cls_ids = (indices // (H * W)).long()
        spatial_indices = indices % (H * W)
        ys = (spatial_indices // W).float()
        xs = (spatial_indices % W).float()

        # 3. Gather regression values
        # regression: (B, 8, H, W) -> (B, 8, H*W)
        reg_flat = regression.view(B, 8, -1)
        # Gather: (B, 8, K)
        # We need to expand indices to (B, 8, K)
        inds_expanded = spatial_indices.unsqueeze(1).expand(-1, 8, -1)
        reg_vals = torch.gather(reg_flat, 2, inds_expanded)  # (B, 8, K)
        reg_vals = reg_vals.permute(0, 2, 1)  # (B, K, 8)

        # 4. Decode Boxes
        # reg_vals: [dx, dy, z, lw, ll, lh, sin, cos]
        # Grid coords

        # We need to map grid (xs, ys) to world coords.
        # Feature map stride relative to input grid?
        # Based on backbone, output stride is 1 relative to input grid (800x800).
        # Input grid voxel size = 0.25.
        # So feature pixel size = 0.25.
        feature_stride = 1
        voxel_size = Config.VOXEL_SIZE
        pc_range = Config.POINT_CLOUD_RANGE

        # Center of pixel in world coords
        xs_world = (
            (xs * feature_stride * voxel_size[0])
            + pc_range[0]
            + (voxel_size[0] * feature_stride / 2)
        )
        ys_world = (
            (ys * feature_stride * voxel_size[1])
            + pc_range[1]
            + (voxel_size[1] * feature_stride / 2)
        )

        # Add offsets
        # reg_vals[:, :, 0] is dx (in meters usually, or relative to stride)
        # CenterPoint usually predicts offset in pixels or meters. Let's assume meters.
        final_x = xs_world + reg_vals[:, :, 0]
        final_y = ys_world + reg_vals[:, :, 1]
        final_z = reg_vals[:, :, 2]

        final_w = torch.exp(reg_vals[:, :, 3])
        final_l = torch.exp(reg_vals[:, :, 4])
        final_h = torch.exp(reg_vals[:, :, 5])

        # Yaw
        sin_y = reg_vals[:, :, 6]
        cos_y = reg_vals[:, :, 7]
        final_yaw = torch.atan2(sin_y, cos_y)

        boxes = torch.stack(
            [final_x, final_y, final_z, final_w, final_l, final_h, final_yaw], dim=-1
        )

        return boxes, scores, cls_ids


class RoIHead(nn.Module):
    """
    Stage 2: RoI Refinement and IoU Rectification.
    """

    def __init__(self, input_channels):
        super().__init__()

        self.roi_size = Config.ROI_SIZE
        self.out_channels = Config.ROI_OUT_CHANNELS

        # Feature Extractor
        self.fc_shared = nn.Sequential(
            nn.Linear(
                input_channels * self.roi_size * self.roi_size, self.out_channels
            ),
            nn.BatchNorm1d(self.out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.out_channels, self.out_channels),
            nn.BatchNorm1d(self.out_channels),
            nn.ReLU(inplace=True),
        )

        # Box Refinement Branch
        # Predicts residuals [dx, dy, dz, dw, dl, dh, dyaw]
        self.reg_branch = nn.Linear(self.out_channels, 7)

        # IoU Rectification Branch
        # Predicts 3D IoU [0, 1]
        self.iou_branch = nn.Sequential(nn.Linear(self.out_channels, 1), nn.Sigmoid())

    def get_roi_features(self, features, boxes):
        """
        Extracts features for rotated boxes using Affine Grid Sampling.
        Args:
            features: (B, C, H, W)
            boxes: (B, K, 7) [x, y, z, w, l, h, yaw]
        Returns:
            roi_features: (B*K, C, ROI_SIZE, ROI_SIZE)
        """
        B, C, H, W = features.shape
        K = boxes.shape[1]

        # Flatten batch
        boxes_flat = boxes.view(-1, 7)  # (B*K, 7)

        # 1. Construct Affine Matrices
        # We map Normalized Target Coords [-1, 1] to Normalized Source Coords [-1, 1]
        # Source limits (World Coords)
        x_min, y_min = Config.POINT_CLOUD_RANGE[0], Config.POINT_CLOUD_RANGE[1]
        x_max, y_max = Config.POINT_CLOUD_RANGE[3], Config.POINT_CLOUD_RANGE[4]
        world_w = x_max - x_min
        world_h = y_max - y_min

        # Box params
        cx, cy = boxes_flat[:, 0], boxes_flat[:, 1]
        w, l = boxes_flat[:, 3], boxes_flat[:, 4]
        yaw = boxes_flat[:, 6]

        # Normalize dimensions to [-1, 1] space
        # A length of 'w' in world space corresponds to w / world_w * 2 in normalized space
        w_norm = w / world_w * 2
        l_norm = l / world_h * 2

        # Normalize centers
        # cx in [x_min, x_max] -> -1 + 2 * (cx - x_min) / world_w
        cx_norm = -1.0 + 2.0 * (cx - x_min) / world_w
        cy_norm = -1.0 + 2.0 * (cy - y_min) / world_h

        # Rotation
        # We want to sample a grid aligned with the box.
        # The affine matrix maps target (grid) to source (image).
        # x_s = R00*x_t + R01*y_t + Tx
        # We want x_t along box width (x-axis local) and y_t along box length (y-axis local).
        # Standard definition: w is x-size, l is y-size.

        cos_a = torch.cos(yaw)
        sin_a = torch.sin(yaw)

        # Note: y-axis in image space (top-down) vs world space.
        # Usually Feature map Y increases downwards?
        # Here we treat feature map as Cartesian grid matching world coordinates (y increases upwards).
        # grid_sample uses (-1, -1) as top-left.
        # If our features are generated such that y-index 0 is y_min (bottom),
        # and grid_sample treats y=-1 as top, we might need to flip.
        # PointPillars Scatter: y_idx 0 corresponds to y_min.
        # Standard image convention: y=0 is top.
        # So y_idx 0 is top in tensor memory if printed as image.
        # But physically it corresponds to y_min.
        # If we use `grid_sample`, y=-1 is top.
        # We need to be careful.
        # Let's assume standard Cartesian mapping where we handled coordinates consistently.
        # If y_idx=0 is y_min, and grid_sample y=-1 is y=0 index (top), then there is a flip.
        # Let's assume standard behavior: y=-1 is index 0.
        # Our y_idx 0 is y_min.
        # So y_min maps to y=-1. y_max maps to y=1.
        # This matches `grid_sample` if we consider the tensor as [y_min ... y_max] from top to bottom?
        # No, usually index 0 is top.
        # If we filled canvas such that y_idx=0 (y_min) is at index 0, then y_min is at "top".
        # So y=-1 corresponds to y_min. This is consistent.

        # Matrix construction
        # Scale factors (half-width because grid is [-1, 1])
        sx = w_norm / 2.0
        sy = l_norm / 2.0

        # R = [ [sx cos, -sy sin], [sx sin, sy cos] ]
        # Wait, if we move 1 unit in x_t, we move sx in x_s (rotated).
        theta = torch.zeros(B * K, 2, 3, device=features.device)
        theta[:, 0, 0] = sx * cos_a
        theta[:, 0, 1] = -sy * sin_a
        theta[:, 0, 2] = cx_norm

        theta[:, 1, 0] = sx * sin_a
        theta[:, 1, 1] = sy * cos_a
        theta[:, 1, 2] = cy_norm

        # Generate Grid
        grid = F.affine_grid(
            theta,
            torch.Size((B * K, C, self.roi_size, self.roi_size)),
            align_corners=True,
        )

        # Sample
        # We need to repeat features for each box? No, too expensive.
        # We must iterate batch.

        roi_features_list = []
        # Reshape grid to (B, K, H, W, 2)
        grid = grid.view(B, K, self.roi_size, self.roi_size, 2)

        for b in range(B):
            # feat: (1, C, H_f, W_f)
            feat_b = features[b : b + 1]
            # grid_b: (K, H_r, W_r, 2) -> Treat K as batch for grid_sample
            grid_b = grid[b]

            # Expand feat to match K? No, grid_sample supports broadcasting?
            # grid_sample(input, grid): input (N, C, Hin, Win), grid (N, Hout, Wout, 2)
            # We have 1 input image, K grids.
            # We must expand input to (K, C, Hin, Win).
            # This is memory heavy if K is large (e.g. 500).
            # 500 * 384 * 800 * 800 * 4 bytes is huge.
            # Efficient way: crop from (1, C, H, W) using (K, H, W, 2) grid?
            # PyTorch `grid_sample` does not support broadcasting input against grid batch.
            # We must loop or chunk.

            # Chunking to save memory
            chunk_size = 64
            for i in range(0, K, chunk_size):
                sub_grid = grid_b[i : i + chunk_size]  # (k, h, w, 2)
                k_curr = sub_grid.shape[0]
                sub_feat = feat_b.expand(
                    k_curr, -1, -1, -1
                )  # Virtual expansion (stride 0)

                # Sampling
                out = F.grid_sample(
                    sub_feat, sub_grid, align_corners=True
                )  # (k, C, h, w)
                roi_features_list.append(out)

        roi_features = torch.cat(roi_features_list, dim=0)  # (B*K, C, H, W)

        return roi_features

    def forward(self, features, proposals):
        """
        Args:
            features: (B, C, H, W)
            proposals: (B, K, 7)
        """
        # 1. Extract RoI Features
        roi_feats = self.get_roi_features(features, proposals)

        # 2. Flatten
        roi_feats_flat = roi_feats.view(roi_feats.shape[0], -1)

        # 3. Shared FC
        feat = self.fc_shared(roi_feats_flat)

        # 4. Heads
        residuals = self.reg_branch(feat)  # (B*K, 7)
        iou_pred = self.iou_branch(feat)  # (B*K, 1)

        # Reshape to (B, K, ...)
        B, K, _ = proposals.shape
        residuals = residuals.view(B, K, 7)
        iou_pred = iou_pred.view(B, K)

        return residuals, iou_pred


class TwoStagePointPillars(nn.Module):
    def __init__(self):
        super().__init__()
        self.vfe = PillarVFE()
        self.scatter = PointPillarsScatter()
        self.backbone = Backbone()
        self.center_head = CenterHead(self.backbone.out_channels)
        self.roi_head = RoIHead(self.backbone.out_channels)

    def forward(self, voxels, num_points, coors, batch_size=None):
        # 1. VFE
        voxel_features = self.vfe(voxels, num_points, coors)

        # 2. Scatter
        if batch_size is None:
            batch_size = int(coors[:, 0].max().item()) + 1
        bev_map = self.scatter(voxel_features, coors, batch_size)

        # 3. Backbone
        x = self.backbone(bev_map)

        # 4. Stage 1: CenterHead
        heatmap, regression = self.center_head(x)

        return heatmap, regression, x

    def forward_stage2(self, x, proposals):
        # x: Backbone features
        # proposals: (B, K, 7)
        residuals, iou_pred = self.roi_head(x, proposals)
        return residuals, iou_pred
