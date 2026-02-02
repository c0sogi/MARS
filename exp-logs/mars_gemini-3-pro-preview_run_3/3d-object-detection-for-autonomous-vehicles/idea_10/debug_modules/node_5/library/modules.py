import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_max
from library.config import Config


class PillarEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.voxel_size = nn.Parameter(
            torch.tensor(Config.VOXEL_SIZE), requires_grad=False
        )
        self.pc_range = nn.Parameter(
            torch.tensor(Config.POINT_CLOUD_RANGE), requires_grad=False
        )
        self.grid_size = nn.Parameter(
            torch.tensor(Config.GRID_SIZE), requires_grad=False
        )

        self.in_channels = Config.IN_CHANNELS
        self.out_channels = Config.PFN_FILTERS[0]

        # PointNet-like feature extractor: Linear -> BN -> ReLU
        self.linear = nn.Linear(self.in_channels, self.out_channels, bias=False)
        self.norm = nn.BatchNorm1d(self.out_channels)

    def forward(self, batched_points):
        """
        Args:
            batched_points: list of (N, 4) tensors [x, y, z, intensity]
        Returns:
            bev_map: (B, C, H, W) tensor
        """
        device = batched_points[0].device
        batch_size = len(batched_points)

        # 1. Prepare batch data
        all_points = []
        batch_indices = []
        for i, points in enumerate(batched_points):
            # Filter points outside range
            mask = (
                (points[:, 0] >= self.pc_range[0])
                & (points[:, 0] < self.pc_range[3])
                & (points[:, 1] >= self.pc_range[1])
                & (points[:, 1] < self.pc_range[4])
                & (points[:, 2] >= self.pc_range[2])
                & (points[:, 2] < self.pc_range[5])
            )
            valid_points = points[mask]

            if valid_points.shape[0] > 0:
                all_points.append(valid_points)
                batch_indices.append(
                    torch.full(
                        (valid_points.shape[0],), i, device=device, dtype=torch.long
                    )
                )

        if not all_points:
            return torch.zeros(
                (
                    batch_size,
                    self.out_channels,
                    int(self.grid_size[1]),
                    int(self.grid_size[0]),
                ),
                device=device,
            )

        all_points = torch.cat(all_points, dim=0)
        batch_indices = torch.cat(batch_indices, dim=0)

        # 2. Voxelization
        # Calculate grid coordinates
        coords = ((all_points[:, :3] - self.pc_range[:3]) / self.voxel_size).long()

        # Clamp to ensure safety (though filter above should handle it)
        coords[:, 0] = torch.clamp(coords[:, 0], 0, int(self.grid_size[0]) - 1)
        coords[:, 1] = torch.clamp(coords[:, 1], 0, int(self.grid_size[1]) - 1)

        # Compute unique pillar keys: b * (H*W) + y * W + x
        # Note: z is ignored for BEV pillarization (z-axis is collapsed)
        H, W = int(self.grid_size[1]), int(self.grid_size[0])
        keys = batch_indices * (H * W) + coords[:, 1] * W + coords[:, 0]

        # 3. Feature Extraction (PFN)
        x = self.linear(all_points)
        x = self.norm(x)
        x = F.relu(x)

        # 4. Scatter Max Pooling
        # Group points by pillar key and take max
        # inverse_indices maps each point to its unique pillar index
        unique_keys, inverse_indices = torch.unique(
            keys, return_inverse=True, sorted=True
        )

        # scatter_max returns (values, indices), we only need values
        pillar_features, _ = scatter_max(x, inverse_indices, dim=0)

        # 5. Scatter to BEV Canvas
        # Create canvas (B * H * W, C)
        canvas = torch.zeros((batch_size * H * W, self.out_channels), device=device)
        canvas[unique_keys] = pillar_features

        # Reshape to (B, C, H, W)
        # Note: canvas is currently (B*H*W, C), need to transpose to (B, H, W, C) then permute
        canvas = canvas.view(batch_size, H, W, self.out_channels)
        canvas = canvas.permute(0, 3, 1, 2).contiguous()

        return canvas


class ResNetFPN(nn.Module):
    def __init__(self):
        super().__init__()

        in_filters = Config.PFN_FILTERS[0]
        layer_filters = Config.LAYER_FILTERS
        layer_strides = Config.LAYER_STRIDES
        upsample_strides = Config.UPSAMPLE_STRIDES
        upsample_filters = Config.NUM_UPSAMPLE_FILTERS

        # Encoder Blocks
        self.blocks = nn.ModuleList()
        current_in = in_filters

        for i, (out_filt, stride) in enumerate(zip(layer_filters, layer_strides)):
            block = nn.Sequential(
                nn.Conv2d(
                    current_in, out_filt, 3, stride=stride, padding=1, bias=False
                ),
                nn.BatchNorm2d(out_filt),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_filt, out_filt, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_filt),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_filt, out_filt, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_filt),
                nn.ReLU(inplace=True),
            )
            self.blocks.append(block)
            current_in = out_filt

        # Decoder / Upsample Blocks
        self.deblocks = nn.ModuleList()
        for i, (in_filt, stride, out_filt) in enumerate(
            zip(layer_filters, upsample_strides, upsample_filters)
        ):
            if stride > 1:
                deblock = nn.Sequential(
                    nn.ConvTranspose2d(
                        in_filt, out_filt, stride, stride=stride, bias=False
                    ),
                    nn.BatchNorm2d(out_filt),
                    nn.ReLU(inplace=True),
                )
            else:
                deblock = nn.Sequential(
                    nn.Conv2d(in_filt, out_filt, 3, padding=1, bias=False),
                    nn.BatchNorm2d(out_filt),
                    nn.ReLU(inplace=True),
                )
            self.deblocks.append(deblock)

    def forward(self, x):
        """
        Returns:
            dict containing:
            'features': Combined FPN features (B, C_out, H, W)
            'spatial_features_2d': List of intermediate features [P1, P2, P3]
        """
        spatial_features = []
        upsampled_features = []

        # Encoder path
        for block in self.blocks:
            x = block(x)
            spatial_features.append(x)

        # Decoder path
        for i, deblock in enumerate(self.deblocks):
            up = deblock(spatial_features[i])
            upsampled_features.append(up)

        # Concat
        combined = torch.cat(upsampled_features, dim=1)

        return {"features": combined, "spatial_features_2d": spatial_features}


class CenterHead(nn.Module):
    def __init__(self):
        super().__init__()

        in_channels = Config.FPN_OUT_CHANNELS
        self.tasks = Config.TASKS
        self.common_heads = Config.COMMON_HEADS

        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=True),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

        self.task_heads = nn.ModuleList()

        for task in self.tasks:
            heads = {}
            # Heatmap head
            heads["hm"] = nn.Sequential(
                nn.Conv2d(in_channels, 64, 3, padding=1, bias=True),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, task["num_class"], 1, bias=True),
            )
            # Initialize bias for heatmap to -2.19 (focal loss trick)
            heads["hm"][-1].bias.data.fill_(-2.19)

            # Regression heads
            for head_name, (out_c, num_conv) in self.common_heads.items():
                heads[head_name] = nn.Sequential(
                    nn.Conv2d(in_channels, 64, 3, padding=1, bias=True),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, out_c, 1, bias=True),
                )

            self.task_heads.append(nn.ModuleDict(heads))

    def forward(self, x):
        x = self.shared_conv(x)
        ret_dicts = []
        for task_head in self.task_heads:
            ret_dict = {}
            for head_name, head_layer in task_head.items():
                ret_dict[head_name] = head_layer(x)
            ret_dicts.append(ret_dict)
        return ret_dicts

    def get_proposals(self, preds_dicts):
        """
        Generate proposals from predictions.
        """
        proposals = []

        for task_idx, preds in enumerate(preds_dicts):
            batch_size = preds["hm"].shape[0]

            # Sigmoid on heatmap
            hm = torch.sigmoid(preds["hm"])

            # Simple NMS via MaxPool
            pad = 1
            hmax = F.max_pool2d(hm, kernel_size=3, stride=1, padding=pad)
            keep = (hmax == hm).float()
            hm = hm * keep

            # Flatten
            hm = hm.view(batch_size, -1)

            # TopK
            K = min(Config.MAX_PROPOSALS, hm.shape[1])
            scores, inds = torch.topk(hm, K)

            # Decode location
            H, W = preds["hm"].shape[2], preds["hm"].shape[3]
            ys = (inds // W).float()
            xs = (inds % W).float()

            # Grid to world
            # x_world = (x_grid * stride + offset) * voxel_size + min_range
            # Stride is 1 relative to the output grid (1920x1920)
            # Center of pixel: +0.5
            xs = xs * Config.VOXEL_SIZE[0] + Config.POINT_CLOUD_RANGE[0]
            ys = ys * Config.VOXEL_SIZE[1] + Config.POINT_CLOUD_RANGE[1]

            # Gather regression heads
            reg = preds["reg"].view(batch_size, 2, -1)  # dx, dy
            height = preds["height"].view(batch_size, 1, -1)  # dz
            dim = preds["dim"].view(batch_size, 3, -1)  # w, l, h
            rot = preds["rot"].view(batch_size, 2, -1)  # sin, cos

            # Gather using inds
            def gather_feat(feat, ind):
                dim = feat.size(1)
                ind = ind.expand(ind.size(0), dim, ind.size(2))
                return feat.gather(2, ind)

            reg = gather_feat(reg, inds.unsqueeze(1))
            height = gather_feat(height, inds.unsqueeze(1))
            dim = gather_feat(dim, inds.unsqueeze(1))
            rot = gather_feat(rot, inds.unsqueeze(1))

            # Add regression offset
            xs = xs + reg[:, 0, :]
            ys = ys + reg[:, 1, :]
            zs = height[:, 0, :]

            # Dimensions (exp)
            dim = torch.exp(dim)

            # Rotation (atan2)
            yaw = torch.atan2(rot[:, 0, :], rot[:, 1, :])

            # Stack: (B, K, 7) -> x, y, z, w, l, h, yaw
            boxes = torch.stack(
                [xs, ys, zs, dim[:, 0, :], dim[:, 1, :], dim[:, 2, :], yaw], dim=2
            )

            # Add scores and class label
            # Class label depends on task.
            # task['class_names'] maps 0..N to specific names.
            # Here we just store the box and score.
            # We need to track batch index for RoI head

            for b in range(batch_size):
                # Filter low scores
                mask = scores[b] > Config.SCORE_THRESHOLD
                if mask.sum() > 0:
                    b_boxes = boxes[b][mask]
                    b_scores = scores[b][mask]

                    # Store as (x, y, z, w, l, h, yaw, score, batch_idx, task_idx)
                    # We append batch_idx to help RoI extraction
                    b_idx_tensor = torch.full(
                        (b_boxes.shape[0], 1), b, device=b_boxes.device
                    )
                    proposals.append(
                        torch.cat([b_boxes, b_scores.unsqueeze(1), b_idx_tensor], dim=1)
                    )

        if len(proposals) == 0:
            return None

        return torch.cat(proposals, dim=0)


class RoIHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.roi_size = Config.ROI_ALIGN_SIZE
        self.out_dim = Config.ROI_HEAD_DIM

        # Multi-scale fusion: 3 levels * 64/128/256 filters?
        # Config.LAYER_FILTERS = [64, 128, 256]
        input_dim = sum(Config.LAYER_FILTERS)  # 64+128+256 = 448

        self.shared_fc = nn.Sequential(
            nn.Linear(input_dim * self.roi_size * self.roi_size, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
        )

        # Refinement Head: Predict residual (dx, dy, dz, dw, dl, dh, dyaw)
        self.reg_head = nn.Linear(256, 7)

        # IoU Head: Predict IoU (0-1)
        self.iou_head = nn.Linear(256, 1)

    def roi_align_rotated(self, features_list, boxes, batch_indices):
        """
        Extract features from multiple FPN levels.
        features_list: [P1, P2, P3]
        boxes: (N, 7) [x, y, z, w, l, h, yaw]
        batch_indices: (N,)
        """
        num_boxes = boxes.shape[0]
        pooled_feats = []

        # Generate canonical grid (N, H, W, 2)
        # Grid range [-1, 1] corresponds to box width/length
        y = torch.linspace(-1, 1, self.roi_size, device=boxes.device)
        x = torch.linspace(-1, 1, self.roi_size, device=boxes.device)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        grid = torch.stack([grid_x, grid_y], dim=-1)  # (7, 7, 2)
        grid = grid.unsqueeze(0).repeat(num_boxes, 1, 1, 1)  # (N, 7, 7, 2)

        # Scale grid by box dimensions (w, l)
        # box: x, y, z, w, l, h, yaw
        w = boxes[:, 3].view(num_boxes, 1, 1, 1)
        l = boxes[:, 4].view(num_boxes, 1, 1, 1)

        # Grid points in local metric coordinates (relative to center)
        # grid vals are -1..1, so multiply by w/2, l/2
        grid_metric = grid * torch.cat([w, l], dim=-1).unsqueeze(2) * 0.5

        # Rotate grid
        yaw = boxes[:, 6]
        c = torch.cos(yaw).view(num_boxes, 1, 1)
        s = torch.sin(yaw).view(num_boxes, 1, 1)

        # x_rot = x*c - y*s
        # y_rot = x*s + y*c
        x_local = grid_metric[..., 0]
        y_local = grid_metric[..., 1]

        x_rot = x_local * c - y_local * s
        y_rot = x_local * s + y_local * c

        # Translate to world coordinates
        x_world = x_rot + boxes[:, 0].view(num_boxes, 1, 1)
        y_world = y_rot + boxes[:, 1].view(num_boxes, 1, 1)

        # Sample from each level
        strides = Config.LAYER_STRIDES  # [1, 2, 4]

        for feat, stride in zip(features_list, strides):
            # Normalize world coords to feature map coords [-1, 1]
            # Feature map extent:
            # Min: RANGE[0], Max: RANGE[3]
            # Size: GRID_SIZE / stride

            x_min, y_min = Config.POINT_CLOUD_RANGE[0], Config.POINT_CLOUD_RANGE[1]
            x_max, y_max = Config.POINT_CLOUD_RANGE[3], Config.POINT_CLOUD_RANGE[4]

            # Normalize to [0, 1]
            u = (x_world - x_min) / (x_max - x_min)
            v = (y_world - y_min) / (y_max - y_min)

            # Normalize to [-1, 1] for grid_sample
            u = 2 * u - 1
            v = 2 * v - 1

            sample_grid = torch.stack([u, v], dim=-1)  # (N, 7, 7, 2)

            # We need to handle batch index. grid_sample takes (B, C, H, W)
            # But we have N boxes from mixed batches.
            # We process one batch item at a time or reshape.
            # Processing per batch item is safer.

            level_samples = []
            unique_batches = torch.unique(batch_indices)

            # Pre-allocate output
            out = torch.zeros(
                (num_boxes, feat.shape[1], self.roi_size, self.roi_size),
                device=feat.device,
            )

            for b in unique_batches:
                mask = batch_indices == b
                if not mask.any():
                    continue

                b_grid = sample_grid[mask].unsqueeze(0)  # (1, N_b, 7, 7, 2)
                # Reshape to (1, N_b*7, 7, 2) to trick grid_sample? No.
                # grid_sample expects (B, H_out, W_out, 2).
                # Here we want to sample at arbitrary points.
                # Treat N_b boxes as the "Height" dimension of the output grid?
                # grid: (1, N_b*7, 7, 2)

                b_grid_flat = b_grid.view(1, -1, self.roi_size, 2)
                b_feat = feat[b : b + 1]  # (1, C, H, W)

                sample = F.grid_sample(
                    b_feat, b_grid_flat, align_corners=False
                )  # (1, C, N_b*7, 7)

                # Reshape back
                sample = sample.view(
                    feat.shape[1], -1, self.roi_size, self.roi_size
                )  # (C, N_b, 7, 7)
                sample = sample.permute(1, 0, 2, 3)  # (N_b, C, 7, 7)

                out[mask] = sample

            pooled_feats.append(out)

        return torch.cat(pooled_feats, dim=1)  # Concat along channel dim

    def forward(self, features_dict, proposals):
        """
        Args:
            features_dict: Output from ResNetFPN
            proposals: (N, 9) [x, y, z, w, l, h, yaw, score, batch_idx]
        """
        if proposals is None or proposals.shape[0] == 0:
            return None

        spatial_feats = features_dict["spatial_features_2d"]
        boxes = proposals[:, :7]
        batch_idx = proposals[:, 8].long()

        # RoI Align
        roi_feats = self.roi_align_rotated(spatial_feats, boxes, batch_idx)

        # Flatten
        x = roi_feats.view(roi_feats.shape[0], -1)

        # MLP
        x = self.shared_fc(x)

        # Heads
        refine = self.reg_head(x)
        iou = torch.sigmoid(self.iou_head(x))

        return refine, iou


class PointPillars(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = PillarEncoder()
        self.backbone = ResNetFPN()
        self.head = CenterHead()
        self.roi_head = RoIHead()

    def forward(self, batched_points):
        # 1. Encode
        bev_map = self.encoder(batched_points)

        # 2. Backbone
        features = self.backbone(bev_map)

        # 3. Stage 1 Head
        preds_dicts = self.head(features["features"])

        # 4. Generate Proposals
        # We need proposals for Stage 2
        # During training, we should technically use both GT and Proposals,
        # but for this implementation we'll assume inference flow or simple proposal-based training
        with torch.no_grad():
            proposals = self.head.get_proposals(preds_dicts)

        # 5. Stage 2 Refinement
        if proposals is not None:
            refine, iou = self.roi_head(features, proposals)

            # Apply refinement to proposals
            # refine is (dx, dy, dz, dw, dl, dh, dyaw)
            # We apply it to the proposals
            refined_boxes = proposals.clone()
            refined_boxes[:, 0] += refine[:, 0]  # x
            refined_boxes[:, 1] += refine[:, 1]  # y
            refined_boxes[:, 2] += refine[:, 2]  # z
            refined_boxes[:, 3] *= torch.exp(refine[:, 3])  # w
            refined_boxes[:, 4] *= torch.exp(refine[:, 4])  # l
            refined_boxes[:, 5] *= torch.exp(refine[:, 5])  # h
            refined_boxes[:, 6] += refine[:, 6]  # yaw

            # Rectify score
            # Final score = Stage1_Score * Predicted_IoU
            rectified_scores = proposals[:, 7] * iou.squeeze()

            return {
                "stage1_preds": preds_dicts,
                "proposals": proposals,
                "refined_boxes": refined_boxes,
                "rectified_scores": rectified_scores,
                "pred_iou": iou,
            }
        else:
            return {
                "stage1_preds": preds_dicts,
                "proposals": None,
                "refined_boxes": None,
                "rectified_scores": None,
                "pred_iou": None,
            }
