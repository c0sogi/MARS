import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from library.config import Config


class PillarVoxelization(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.voxel_size = torch.tensor(config.VOXEL_SIZE).float()
        self.pc_range = torch.tensor(config.PC_RANGE).float()
        self.max_points = config.MAX_POINTS_PER_PILLAR
        self.max_pillars = config.MAX_PILLARS_TRAIN

        # Calculate Grid Size
        # (51.2 - (-51.2)) / 0.2 = 512
        self.grid_size = (self.pc_range[3:] - self.pc_range[:3]) / self.voxel_size
        self.grid_size = self.grid_size.round().long()
        self.nx, self.ny = self.grid_size[0].item(), self.grid_size[1].item()

    def forward(self, batched_points):
        """
        Args:
            batched_points: List[Tensor(N, 4)]
        Returns:
            features: (P, max_points, 9)
            coords: (P, 4) [batch_idx, z, y, x]
        """
        pillars_list = []
        coors_list = []

        device = batched_points[0].device
        self.voxel_size = self.voxel_size.to(device)
        self.pc_range = self.pc_range.to(device)

        for batch_idx, points in enumerate(batched_points):
            if points.shape[0] == 0:
                continue

            # 1. Filter points outside range
            mask = (
                (points[:, 0] >= self.pc_range[0])
                & (points[:, 0] < self.pc_range[3])
                & (points[:, 1] >= self.pc_range[1])
                & (points[:, 1] < self.pc_range[4])
                & (points[:, 2] >= self.pc_range[2])
                & (points[:, 2] < self.pc_range[5])
            )
            points = points[mask]

            if points.shape[0] == 0:
                continue

            # 2. Calculate Grid Coordinates
            # (N, 3) -> (x_idx, y_idx, z_idx)
            coords = ((points[:, :3] - self.pc_range[:3]) / self.voxel_size).long()

            # 3. Assign to Pillars (1D Key)
            # key = y * nx + x
            keys = coords[:, 1] * self.nx + coords[:, 0]

            # 4. Sort points by key to group them
            sorted_idx = torch.argsort(keys)
            points_sorted = points[sorted_idx]
            keys_sorted = keys[sorted_idx]

            # 5. Identify unique pillars and counts
            unique_keys, counts = torch.unique_consecutive(
                keys_sorted, return_counts=True
            )

            # 6. Limit number of pillars
            if unique_keys.size(0) > self.max_pillars:
                unique_keys = unique_keys[: self.max_pillars]
                counts = counts[: self.max_pillars]
                # Slice points: sum of counts is the number of points to keep
                total_points = counts.sum().item()
                points_sorted = points_sorted[:total_points]
                keys_sorted = keys_sorted[
                    :total_points
                ]  # Not strictly needed but good for consistency

            num_pillars = unique_keys.size(0)

            # 7. Create Dense Pillar Tensor
            # We need to map each point to (pillar_idx, point_idx_in_pillar)

            # pillar_idx: repeat pillar index by count
            # [0, 0, 1, 1, 1, ...]
            pillar_indices = torch.repeat_interleave(
                torch.arange(num_pillars, device=device), counts
            )

            # point_idx_in_pillar:
            # We construct offsets: [0, count0, count0+count1, ...]
            # Then point_idx = arange(N) - offsets[pillar_indices]
            offsets = torch.cat(
                [torch.tensor([0], device=device), counts.cumsum(0)[:-1]]
            )
            point_indices = (
                torch.arange(points_sorted.shape[0], device=device)
                - offsets[pillar_indices]
            )

            # 8. Filter points exceeding max_points_per_pillar
            mask_points = point_indices < self.max_points

            points_final = points_sorted[mask_points]
            p_idx = pillar_indices[mask_points]
            n_idx = point_indices[mask_points]

            # Initialize feature tensor: (P, N, 9)
            # 9 features: x, y, z, i, x_c, y_c, z_c, x_p, y_p
            pillar_features = torch.zeros(
                (num_pillars, self.max_points, 9), device=device
            )

            # Fill raw features (x, y, z, i)
            pillar_features[p_idx, n_idx, :4] = points_final

            # 9. Feature Decoration
            # Calculate geometric centers (x_c, y_c, z_c)
            x_idx = unique_keys % self.nx
            y_idx = unique_keys // self.nx

            x_c = (
                x_idx.float() * self.voxel_size[0]
                + self.pc_range[0]
                + self.voxel_size[0] / 2
            )
            y_c = (
                y_idx.float() * self.voxel_size[1]
                + self.pc_range[1]
                + self.voxel_size[1] / 2
            )
            z_c = torch.full_like(x_c, self.pc_range[2] + self.voxel_size[2] / 2)

            centers = torch.stack([x_c, y_c, z_c], dim=1).unsqueeze(1)  # (P, 1, 3)

            # Mask for valid points
            valid_mask = (
                (pillar_features[:, :, 0] != 0)
                | (pillar_features[:, :, 1] != 0)
                | (pillar_features[:, :, 2] != 0)
            )
            valid_mask = valid_mask.unsqueeze(-1)

            # Offset from geometric center
            pillar_features[:, :, 4:7] = (
                pillar_features[:, :, :3] - centers
            ) * valid_mask

            # Offset from arithmetic mean (x_p, y_p, z_p)
            pillar_sum = pillar_features[:, :, :3].sum(dim=1, keepdim=True)
            pillar_counts = valid_mask.sum(dim=1, keepdim=True).float().clamp(min=1.0)
            pillar_means = pillar_sum / pillar_counts

            pillar_features[:, :, 7:9] = (
                pillar_features[:, :, :2] - pillar_means[:, :, :2]
            ) * valid_mask

            pillars_list.append(pillar_features)

            # 10. Coordinates [batch_idx, z, y, x]
            b_idx = torch.full(
                (num_pillars,), batch_idx, device=device, dtype=torch.long
            )
            z_idx = torch.zeros((num_pillars,), device=device, dtype=torch.long)
            batch_coords = torch.stack([b_idx, z_idx, y_idx, x_idx], dim=1)
            coors_list.append(batch_coords)

        if not pillars_list:
            return None, None

        features = torch.cat(pillars_list, dim=0)
        coords = torch.cat(coors_list, dim=0)

        return features, coords


class PillarFeatureNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.in_channels = config.NUM_PILLAR_FEATURES
        self.out_channels = config.PILLAR_FEATURE_NET_FILTERS[0]

        self.linear = nn.Linear(self.in_channels, self.out_channels, bias=False)
        self.bn = nn.BatchNorm1d(self.out_channels)

    def forward(self, features):
        # features: (P, N, 9)
        P, N, D = features.shape

        # Linear: (P*N, 9) -> (P*N, 64)
        x = self.linear(features.view(-1, D))
        x = self.bn(x)
        x = F.relu(x)

        # Reshape: (P, N, 64)
        x = x.view(P, N, -1)

        # Max Pooling: (P, 64)
        x = x.max(dim=1)[0]

        return x


class PointPillarsScatter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.nx = int((config.PC_RANGE[3] - config.PC_RANGE[0]) / config.VOXEL_SIZE[0])
        self.ny = int((config.PC_RANGE[4] - config.PC_RANGE[1]) / config.VOXEL_SIZE[1])
        self.num_channels = config.PILLAR_FEATURE_NET_FILTERS[0]

    def forward(self, pillar_features, coords, batch_size):
        # pillar_features: (P, C)
        # coords: (P, 4) [b, z, y, x]

        canvas = torch.zeros(
            (batch_size, self.num_channels, self.ny, self.nx),
            dtype=pillar_features.dtype,
            device=pillar_features.device,
        )

        b = coords[:, 0]
        y = coords[:, 2]
        x = coords[:, 3]

        canvas[b, :, y, x] = pillar_features

        return canvas


class Backbone(nn.Module):
    def __init__(self, config):
        super().__init__()

        layer_nums = config.BACKBONE_LAYER_NUMS
        layer_strides = config.BACKBONE_LAYER_STRIDES
        num_filters = config.BACKBONE_FILTERS
        upsample_strides = [1, 2, 4]
        num_upsample_filters = config.NECK_FILTERS

        input_channels = config.PILLAR_FEATURE_NET_FILTERS[0]

        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()

        for idx in range(len(layer_nums)):
            block_layers = []
            cur_layers = layer_nums[idx]
            stride = layer_strides[idx]
            out_channels = num_filters[idx]

            # Downsample Layer
            block_layers.append(
                nn.Conv2d(
                    input_channels,
                    out_channels,
                    3,
                    stride=stride,
                    padding=1,
                    bias=False,
                )
            )
            block_layers.append(nn.BatchNorm2d(out_channels))
            block_layers.append(nn.ReLU(inplace=True))

            input_channels = out_channels

            for _ in range(cur_layers - 1):
                block_layers.append(
                    nn.Conv2d(
                        input_channels, out_channels, 3, stride=1, padding=1, bias=False
                    )
                )
                block_layers.append(nn.BatchNorm2d(out_channels))
                block_layers.append(nn.ReLU(inplace=True))

            self.blocks.append(nn.Sequential(*block_layers))

            # Upsample Layer
            up_stride = upsample_strides[idx]
            up_filters = num_upsample_filters[idx]

            if up_stride > 1:
                self.deblocks.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(
                            out_channels,
                            up_filters,
                            up_stride,
                            stride=up_stride,
                            bias=False,
                        ),
                        nn.BatchNorm2d(up_filters),
                        nn.ReLU(inplace=True),
                    )
                )
            else:
                self.deblocks.append(
                    nn.Sequential(
                        nn.Conv2d(
                            out_channels, up_filters, 3, stride=1, padding=1, bias=False
                        ),
                        nn.BatchNorm2d(up_filters),
                        nn.ReLU(inplace=True),
                    )
                )

    def forward(self, x):
        ups = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            ups.append(self.deblocks[i](x))

        if len(ups) > 1:
            return torch.cat(ups, dim=1)
        return ups[0]


class SSDHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_classes = config.NUM_CLASSES
        self.num_anchors = len(config.CLASS_NAMES) * len(config.ANCHOR_ROTATIONS)

        in_channels = sum(config.NECK_FILTERS)

        # Classification: (B, Num_Anchors * Num_Classes, H, W)
        self.cls_conv = nn.Conv2d(in_channels, self.num_anchors * self.num_classes, 1)

        # Regression: (B, Num_Anchors * 7, H, W)
        self.reg_conv = nn.Conv2d(in_channels, self.num_anchors * 7, 1)

        # Initialization
        prior_prob = 0.01
        bias_init = -math.log((1 - prior_prob) / prior_prob)
        self.cls_conv.bias.data.fill_(bias_init)

    def forward(self, x):
        cls_preds = self.cls_conv(x)
        reg_preds = self.reg_conv(x)

        # Permute to (B, H, W, Num_Anchors, ...)
        B, _, H, W = cls_preds.shape

        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        cls_preds = cls_preds.view(B, H, W, self.num_anchors, self.num_classes)

        reg_preds = reg_preds.permute(0, 2, 3, 1).contiguous()
        reg_preds = reg_preds.view(B, H, W, self.num_anchors, 7)

        return cls_preds, reg_preds


class PointPillars(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config if config is not None else Config

        self.voxel_layer = PillarVoxelization(self.config)
        self.pillar_net = PillarFeatureNet(self.config)
        self.scatter = PointPillarsScatter(self.config)
        self.backbone = Backbone(self.config)
        self.head = SSDHead(self.config)

    def forward(self, batched_points, batched_tokens=None):
        # batched_points: List[Tensor(N, 4)]
        batch_size = len(batched_points)

        # 1. Voxelization
        features, coords = self.voxel_layer(batched_points)

        if features is None:
            return None, None

        # 2. Feature Extraction
        pillar_features = self.pillar_net(features)

        # 3. Scatter to Pseudo-Image
        spatial_features = self.scatter(pillar_features, coords, batch_size)

        # 4. Backbone (2D CNN)
        neck_features = self.backbone(spatial_features)

        # 5. Detection Head
        cls_preds, reg_preds = self.head(neck_features)

        return cls_preds, reg_preds
