import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from library.config import Config


class PillarFeatureNet(nn.Module):
    """
    Converts raw point cloud data in pillars to feature vectors.
    """

    def __init__(
        self, num_input_features=5, num_filters=64, voxel_size=None, pc_range=None
    ):
        super(PillarFeatureNet, self).__init__()
        self.name = "PillarFeatureNet"

        # 5 input features + 3 (offset from mean) + 2 (offset from pillar center) = 10
        self.num_input_features = num_input_features
        self.num_output_features = num_filters
        self.num_feature_inputs = num_input_features + 5

        self.voxel_size = voxel_size
        self.pc_range = pc_range

        # Simple Linear -> BN -> ReLU structure
        self.linear = nn.Linear(self.num_feature_inputs, num_filters, bias=False)
        self.norm = nn.BatchNorm1d(num_filters)

    def forward(self, features, num_points, coords):
        """
        Args:
            features: (N, max_points, 5) [x, y, z, i, dt]
            num_points: (N,)
            coords: (N, 4) [batch, z, y, x]
        Returns:
            pillar_features: (N, C)
        """
        # 1. Calculate Arithmetic Mean of points in each pillar
        # Create mask for valid points
        dtype = features.dtype
        device = features.device

        # (N, max_points)
        mask = torch.arange(features.shape[1], device=device).unsqueeze(
            0
        ) < num_points.unsqueeze(1)
        mask = mask.type(dtype).unsqueeze(2)  # (N, P, 1)

        # Sum valid points and divide by count (clamp count to 1 to avoid div by zero)
        masked_features = features * mask
        pillar_sum = masked_features.sum(dim=1, keepdim=True)
        pillar_count = num_points.type(dtype).view(-1, 1, 1).clamp(min=1.0)
        pillar_mean = pillar_sum / pillar_count  # (N, 1, 5)

        # 2. Calculate Offsets
        # f_cluster: offset from arithmetic mean
        f_cluster = features[:, :, :3] - pillar_mean[:, :, :3]

        # f_center: offset from pillar center
        # coords is (batch, z, y, x)
        # x_idx = coords[:, 3], y_idx = coords[:, 2]
        x_idx = coords[:, 3].type(dtype)
        y_idx = coords[:, 2].type(dtype)

        # Calculate physical center of the pillar
        # center = index * size + min + size/2
        f_center_x = (
            x_idx * self.voxel_size[0] + self.pc_range[0] + self.voxel_size[0] / 2
        )
        f_center_y = (
            y_idx * self.voxel_size[1] + self.pc_range[1] + self.voxel_size[1] / 2
        )

        # Expand to (N, P, 1)
        f_center_x = f_center_x.view(-1, 1, 1).expand(-1, features.shape[1], 1)
        f_center_y = f_center_y.view(-1, 1, 1).expand(-1, features.shape[1], 1)

        # Calculate offset
        f_center_xp = features[:, :, 0:1] - f_center_x
        f_center_yp = features[:, :, 1:2] - f_center_y

        # 3. Concatenate Features
        # [original_features, f_cluster, f_center]
        features_ls = [features, f_cluster, f_center_xp, f_center_yp]
        stacked_features = torch.cat(features_ls, dim=-1)  # (N, P, 10)

        # 4. Forward Pass
        # Flatten for Linear Layer: (N*P, 10)
        input_features = stacked_features.view(-1, self.num_feature_inputs)

        x = self.linear(input_features)
        x = self.norm(x)
        x = F.relu(x)

        # Reshape back to (N, P, C)
        x = x.view(stacked_features.shape[0], stacked_features.shape[1], -1)

        # 5. Max Pooling over points
        # Apply mask again to ensure padding doesn't affect max
        # However, ReLU makes everything >= 0. If padding is 0, it's fine.
        # But if features can be negative before ReLU? No, ReLU is last.
        # But padding in input was 0. Linear(0) -> 0 (no bias). BN(0) -> beta (if 0 is mean).
        # So padding might become non-zero.
        # We should mask the output of ReLU or use max over masked values.
        # Simpler approach: replace invalid points with -inf before max
        # But standard implementation usually relies on max(0, x) being safe enough if padding is handled.
        # Let's use the mask to zero out invalid points after activation just in case.
        x = x * mask

        # Max pool
        x_max = torch.max(x, dim=1)[0]  # (N, C)

        return x_max


class PointPillarsScatter(nn.Module):
    """
    Scatters pillar features into a 2D pseudo-image.
    """

    def __init__(self, num_input_features=64, grid_size=None):
        super(PointPillarsScatter, self).__init__()
        self.num_channels = num_input_features
        self.nx, self.ny, self.nz = grid_size

    def forward(self, voxel_features, coords, batch_size):
        """
        Args:
            voxel_features: (N, C)
            coords: (N, 4) [batch, z, y, x]
            batch_size: int
        Returns:
            batch_canvas: (B, C, H, W)
        """
        # Create dense canvas
        canvas = torch.zeros(
            batch_size,
            self.num_channels,
            self.ny * self.nx,
            dtype=voxel_features.dtype,
            device=voxel_features.device,
        )

        # Unpack coordinates
        batch_mask = coords[:, 0]
        # Calculate linear index in the spatial grid: y * W + x
        this_coords = coords[:, 2] * self.nx + coords[:, 3]
        this_coords = this_coords.long()

        # Scatter
        # canvas[b, :, flat_idx] = feature
        canvas[batch_mask.long(), :, this_coords] = voxel_features

        # Reshape to (B, C, H, W)
        return canvas.view(batch_size, self.num_channels, self.ny, self.nx)


class Backbone(nn.Module):
    """
    2D Backbone (RPN/FPN style) for feature extraction.
    """

    def __init__(
        self,
        input_channels,
        layer_nums,
        layer_strides,
        num_filters,
        upsample_strides,
        num_upsample_filters,
    ):
        super(Backbone, self).__init__()

        self.layer_nums = layer_nums
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()

        # Downsampling path
        c_in = input_channels
        for i in range(len(layer_nums)):
            block_ops = []
            c_out = num_filters[i]
            stride = layer_strides[i]

            # First layer in block handles stride
            block_ops.append(
                nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False)
            )
            block_ops.append(nn.BatchNorm2d(c_out))
            block_ops.append(nn.ReLU(inplace=True))

            # Subsequent layers
            for _ in range(layer_nums[i] - 1):
                block_ops.append(
                    nn.Conv2d(c_out, c_out, 3, stride=1, padding=1, bias=False)
                )
                block_ops.append(nn.BatchNorm2d(c_out))
                block_ops.append(nn.ReLU(inplace=True))

            self.blocks.append(nn.Sequential(*block_ops))
            c_in = c_out

        # Upsampling path
        for i in range(len(layer_nums)):
            c_in = num_filters[i]
            c_out = num_upsample_filters[i]
            stride = upsample_strides[i]

            self.deblocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(c_in, c_out, stride, stride=stride, bias=False),
                    nn.BatchNorm2d(c_out),
                    nn.ReLU(inplace=True),
                )
            )

        self.num_out_channels = sum(num_upsample_filters)

    def forward(self, x):
        ups = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            ups.append(self.deblocks[i](x))

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        else:
            x = ups[0]

        return x


class CenterHead(nn.Module):
    """
    Anchor-free head for CenterPoint.
    Outputs: Heatmap, Offset, Height, Dim, Rot.
    """

    def __init__(self, in_channels, head_channels, tasks):
        super(CenterHead, self).__init__()

        self.tasks = tasks
        self.heads = nn.ModuleDict()

        for name, out_channels in tasks.items():
            # Standard head design: Conv(64) -> BN -> ReLU -> Conv(out)
            head_layers = [
                nn.Conv2d(
                    in_channels, head_channels, kernel_size=3, padding=1, bias=True
                ),
                nn.BatchNorm2d(head_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    head_channels, out_channels, kernel_size=1, stride=1, padding=0
                ),
            ]

            # Initialize heatmap bias for Focal Loss
            if name == "heatmap":
                last_conv = head_layers[-1]
                # bias = -log((1 - p) / p) with p=0.1 -> approx -2.19
                last_conv.bias.data.fill_(-2.19)

            self.heads[name] = nn.Sequential(*head_layers)

    def forward(self, x):
        ret = {}
        for name, head in self.heads.items():
            ret[name] = head(x)
        return ret


class TemporalPointPillars(nn.Module):
    """
    End-to-End Temporal PointPillars Model.
    """

    def __init__(self):
        super(TemporalPointPillars, self).__init__()
        self.config = Config

        # 1. Pillar Feature Net
        self.pfn = PillarFeatureNet(
            num_input_features=self.config.NUM_POINT_FEATURES,
            num_filters=self.config.PILLAR_FEATURE_NET_FILTERS[0],
            voxel_size=self.config.VOXEL_SIZE,
            pc_range=self.config.POINT_CLOUD_RANGE,
        )

        # 2. Scatter
        grid_size = self.config.get_grid_size()  # [W, H, D]
        self.scatter = PointPillarsScatter(
            num_input_features=self.config.PILLAR_FEATURE_NET_FILTERS[0],
            grid_size=grid_size,
        )

        # 3. Backbone
        self.backbone = Backbone(
            input_channels=self.config.BACKBONE_IN_CHANNELS,
            layer_nums=self.config.BACKBONE_LAYER_NUMS,
            layer_strides=self.config.BACKBONE_LAYER_STRIDES,
            num_filters=self.config.BACKBONE_NUM_FILTERS,
            upsample_strides=self.config.BACKBONE_UPSAMPLE_STRIDES,
            num_upsample_filters=self.config.BACKBONE_NUM_UPSAMPLE_FILTERS,
        )

        # 4. Head
        self.head = CenterHead(
            in_channels=self.backbone.num_out_channels,
            head_channels=self.config.HEAD_CHANNELS,
            tasks=self.config.HEAD_TASKS,
        )

    def forward(self, voxels, num_points, coordinates, **kwargs):
        """
        Args:
            voxels: (M, max_points, 5)
            num_points: (M,)
            coordinates: (M, 4) [batch_idx, z, y, x]
        """
        batch_size = coordinates[:, 0].max().item() + 1

        # 1. PFN
        pillar_features = self.pfn(voxels, num_points, coordinates)

        # 2. Scatter
        spatial_features = self.scatter(pillar_features, coordinates, batch_size)

        # 3. Backbone
        neck_features = self.backbone(spatial_features)

        # 4. Head
        preds = self.head(neck_features)

        return preds
