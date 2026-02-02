import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_max
from library.config import Config

# ==============================================================================
# 1. Pillar Encoder
# ==============================================================================


class PillarEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.voxel_size = Config.VOXEL_SIZE
        self.pc_range = Config.POINT_CLOUD_RANGE
        self.out_dim = Config.PILLAR_FEATURE_DIM

        # Grid dimensions
        self.nx = int((self.pc_range[3] - self.pc_range[0]) / self.voxel_size[0])
        self.ny = int((self.pc_range[4] - self.pc_range[1]) / self.voxel_size[1])

        # Input features: x, y, z, i
        self.in_channels = 4
        # Aggregation: Mean (4) + Max (4) = 8
        self.agg_channels = self.in_channels * 2

        self.encoder = nn.Sequential(
            nn.Linear(self.agg_channels, self.out_dim),
            nn.BatchNorm1d(self.out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, points_list):
        """
        Args:
            points_list: List of tensors, each (N_points, 4) [x, y, z, i]
        Returns:
            bev_map: (B, C, H, W)
        """
        batch_size = len(points_list)
        device = points_list[0].device

        # Containers for batch data
        batch_indices = []
        batch_features = []

        for b_idx, points in enumerate(points_list):
            if points.shape[0] == 0:
                continue

            # Filter out of range points
            mask = (
                (points[:, 0] >= self.pc_range[0])
                & (points[:, 0] < self.pc_range[3])
                & (points[:, 1] >= self.pc_range[1])
                & (points[:, 1] < self.pc_range[4])
                & (points[:, 2] >= self.pc_range[2])
                & (points[:, 2] < self.pc_range[5])
            )
            valid_points = points[mask]

            if valid_points.shape[0] == 0:
                continue

            # Calculate grid coords
            # x_idx = floor((x - x_min) / x_size)
            x_idx = (
                (valid_points[:, 0] - self.pc_range[0]) / self.voxel_size[0]
            ).long()
            y_idx = (
                (valid_points[:, 1] - self.pc_range[1]) / self.voxel_size[1]
            ).long()

            # Clamp to be safe (though mask should handle it)
            x_idx = torch.clamp(x_idx, 0, self.nx - 1)
            y_idx = torch.clamp(y_idx, 0, self.ny - 1)

            # Unique pillar id: b * (ny * nx) + y * nx + x
            # We process batch items together in a flat manner for scatter
            # But here we need to scatter back to (B, C, H, W)
            # Let's create a batch-aware pillar index for the flat batch list
            # We will reconstruct the batch dimension later.

            # Local pillar index within the grid
            pillar_idx = y_idx * self.nx + x_idx

            # Append to lists
            # We need to distinguish batches.
            # Strategy: Offset pillar_idx by batch_idx * (nx * ny)
            batch_offset = b_idx * (self.nx * self.ny)
            global_pillar_idx = pillar_idx + batch_offset

            batch_indices.append(global_pillar_idx)
            batch_features.append(valid_points)

        if not batch_indices:
            return torch.zeros(
                (batch_size, self.out_dim, self.ny, self.nx), device=device
            )

        # Concatenate all batch data
        all_indices = torch.cat(batch_indices)
        all_features = torch.cat(batch_features)

        # Scatter Reduce
        # We need unique pillars to map back to grid
        # But torch_scatter handles duplicates by reducing

        # 1. Mean
        feat_mean = scatter_mean(all_features, all_indices, dim=0)

        # 2. Max
        feat_max, _ = scatter_max(all_features, all_indices, dim=0)

        # Handle empty pillars (scatter_max returns 0 for empty, but we might want min value or 0)
        # Since we only scatter to indices that exist, the rows in feat_mean/max corresponding to
        # existing pillars are valid. Rows for non-existing pillars will be 0.

        # Concatenate
        feat_agg = torch.cat([feat_mean, feat_max], dim=1)

        # MLP
        feat_encoded = self.encoder(
            feat_agg
        )  # (Total_Pillars_Possible_Across_Batch, C_out)

        # Reshape to BEV
        # feat_encoded has shape (B * H * W, C)
        # We need (B, C, H, W)

        # The scatter operation automatically expanded to max index found.
        # We need to ensure it covers the full batch grid size.
        total_pixels = batch_size * self.nx * self.ny

        if feat_encoded.shape[0] < total_pixels:
            padding = torch.zeros(
                (total_pixels - feat_encoded.shape[0], self.out_dim), device=device
            )
            feat_encoded = torch.cat([feat_encoded, padding], dim=0)
        elif feat_encoded.shape[0] > total_pixels:
            # Should not happen if indices are correct
            feat_encoded = feat_encoded[:total_pixels]

        bev_map = feat_encoded.view(batch_size, self.ny, self.nx, self.out_dim)
        bev_map = bev_map.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)

        return bev_map


# ==============================================================================
# 2. DLA-34 Backbone
# ==============================================================================


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            bias=False,
            dilation=dilation,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=dilation,
            bias=False,
            dilation=dilation,
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class Root(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, residual):
        super(Root, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            1,
            stride=1,
            bias=False,
            padding=(kernel_size - 1) // 2,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *x):
        children = x
        x = self.conv(torch.cat(x, 1))
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)

        return x


class Tree(nn.Module):
    def __init__(
        self,
        levels,
        block,
        in_channels,
        out_channels,
        stride=1,
        level_root=False,
        root_dim=0,
        root_kernel_size=1,
        dilation=1,
        root_residual=False,
    ):
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride, dilation=dilation)
            self.tree2 = block(out_channels, out_channels, 1, dilation=dilation)
        else:
            self.tree1 = Tree(
                levels - 1,
                block,
                in_channels,
                out_channels,
                stride,
                root_dim=0,
                root_kernel_size=root_kernel_size,
                dilation=dilation,
                root_residual=root_residual,
            )
            self.tree2 = Tree(
                levels - 1,
                block,
                out_channels,
                out_channels,
                root_dim=root_dim + out_channels,
                root_kernel_size=root_kernel_size,
                dilation=dilation,
                root_residual=root_residual,
            )
        if levels == 1:
            self.root = Root(root_dim, out_channels, root_kernel_size, root_residual)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)
        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=1, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


class DLA34(nn.Module):
    def __init__(
        self,
        in_channels=64,
        levels=[1, 1, 1, 2, 2, 1],
        channels=[16, 32, 64, 128, 256, 512],
    ):
        super(DLA34, self).__init__()
        self.channels = channels

        # Base layer to match input channels
        self.base_layer = nn.Sequential(
            nn.Conv2d(
                in_channels, channels[0], kernel_size=7, stride=1, padding=3, bias=False
            ),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
        )

        self.level0 = self._make_conv_level(channels[0], channels[0], levels[0])
        self.level1 = self._make_conv_level(
            channels[0], channels[1], levels[1], stride=2
        )
        self.level2 = Tree(
            levels[2], BasicBlock, channels[1], channels[2], 2, level_root=False
        )
        self.level3 = Tree(
            levels[3], BasicBlock, channels[2], channels[3], 2, level_root=True
        )
        self.level4 = Tree(
            levels[4], BasicBlock, channels[3], channels[4], 2, level_root=True
        )
        self.level5 = Tree(
            levels[5], BasicBlock, channels[4], channels[5], 2, level_root=True
        )

    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend(
                [
                    nn.Conv2d(
                        inplanes,
                        planes,
                        kernel_size=3,
                        stride=stride if i == 0 else 1,
                        padding=dilation,
                        bias=False,
                        dilation=dilation,
                    ),
                    nn.BatchNorm2d(planes),
                    nn.ReLU(inplace=True),
                ]
            )
            inplanes = planes
        return nn.Sequential(*modules)

    def forward(self, x):
        y = []
        x = self.base_layer(x)
        for i in range(6):
            if i == 0:
                x = self.level0(x)
            elif i == 1:
                x = self.level1(x)
            elif i == 2:
                x = self.level2(x)
            elif i == 3:
                x = self.level3(x)
            elif i == 4:
                x = self.level4(x)
            elif i == 5:
                x = self.level5(x)
            y.append(x)
        return y


# ==============================================================================
# 3. DLA Up (IDAUp)
# ==============================================================================


class IDAUp(nn.Module):
    def __init__(self, node_kernel, out_dim, channels, up_factors):
        super(IDAUp, self).__init__()
        self.channels = channels
        self.out_dim = out_dim
        for i, c in enumerate(channels):
            if c == out_dim:
                proj = nn.Identity()
            else:
                proj = nn.Sequential(
                    nn.Conv2d(c, out_dim, kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(out_dim),
                    nn.ReLU(inplace=True),
                )
            f = int(up_factors[i])
            if f == 1:
                up = nn.Identity()
            else:
                up = nn.ConvTranspose2d(
                    out_dim,
                    out_dim,
                    f * 2,
                    stride=f,
                    padding=f // 2,
                    output_padding=0,
                    groups=out_dim,
                    bias=False,
                )
                # Initialize weights for bilinear interpolation
                self._init_bilinear(up.weight)

            setattr(self, "proj_" + str(i), proj)
            setattr(self, "up_" + str(i), up)

        for i in range(1, len(channels)):
            node = nn.Sequential(
                nn.Conv2d(
                    out_dim * 2,
                    out_dim,
                    kernel_size=node_kernel,
                    stride=1,
                    padding=node_kernel // 2,
                    bias=False,
                ),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(inplace=True),
            )
            setattr(self, "node_" + str(i), node)

    def _init_bilinear(self, weights):
        # Initialize transpose conv to act like bilinear upsampling
        w = weights.size(2)
        c = weights.size(0)
        k = (1.0 / math.ceil(w / 2)) * torch.tensor(
            list(range(1, math.ceil(w / 2) + 1))
            + list(range(math.ceil(w / 2) - 1, 0, -1))
        )
        k = k / k.sum()
        k = torch.outer(k, k).view(1, 1, w, w)
        k = k.repeat(c, 1, 1, 1)
        with torch.no_grad():
            weights.copy_(k)

    def forward(self, layers):
        layers = list(layers)
        for i, l in enumerate(layers):
            upsample = getattr(self, "up_" + str(i))
            project = getattr(self, "proj_" + str(i))
            layers[i] = upsample(project(l))

        x = layers[0]
        y = []
        for i in range(1, len(layers)):
            node = getattr(self, "node_" + str(i))
            x = node(torch.cat([x, layers[i]], 1))
            y.append(x)
        return x


class DLAUp(nn.Module):
    def __init__(self, channels, scales=(1, 2, 4, 8, 16, 32), in_channels=None):
        super(DLAUp, self).__init__()
        # We want to aggregate features to stride 4
        # DLA outputs:
        # level0: stride 1
        # level1: stride 2
        # level2: stride 4
        # level3: stride 8
        # level4: stride 16
        # level5: stride 32

        # We use levels 2, 3, 4, 5
        self.first_level = 2
        self.last_level = 5
        channels = channels[self.first_level :]
        scales = scales[self.first_level :]

        self.out_channels = channels[0]  # The channel count at stride 4

        # IDAUp aggregates from deep to shallow
        # But here we want to aggregate everything to level 2 (stride 4)
        # We can use a simplified IDAUp that upsamples 3->2, 4->2, 5->2 and fuses

        self.ida = IDAUp(
            3, self.out_channels, channels, [2**i for i in range(len(channels))]
        )

    def forward(self, x):
        # x is list of features from DLA
        layers = x[self.first_level : self.last_level + 1]
        out = self.ida(layers)
        return out


# ==============================================================================
# 4. Center Head
# ==============================================================================


class CenterHead(nn.Module):
    def __init__(self, in_channels, heads, head_conv=64):
        super(CenterHead, self).__init__()
        self.heads = heads
        for head, out_c in heads.items():
            fc = nn.Sequential(
                nn.Conv2d(in_channels, head_conv, kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    head_conv, out_c, kernel_size=1, stride=1, padding=0, bias=True
                ),
            )

            # Init
            if "hm" in head:
                fc[-1].bias.data.fill_(-2.19)
            else:
                nn.init.normal_(fc[-1].weight, std=0.001)
                nn.init.constant_(fc[-1].bias, 0)

            self.__setattr__(head, fc)

    def forward(self, x):
        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(x)
        return ret


# ==============================================================================
# 5. IoU-Aware DLA-CenterPoint (Main Module)
# ==============================================================================


class IoUAwareCenterPoint(nn.Module):
    def __init__(self):
        super().__init__()

        # 1. Encoder
        self.encoder = PillarEncoder()

        # 2. Backbone
        # DLA34 channels: [16, 32, 64, 128, 256, 512]
        self.backbone = DLA34(in_channels=Config.PILLAR_FEATURE_DIM)

        # 3. Upsampler
        # We aggregate levels 2,3,4,5 to level 2 (stride 4)
        # Level 2 has 64 channels
        self.neck = DLAUp(channels=self.backbone.channels)

        # 4. Head
        # Input to head is 64 channels (from level 2)
        self.head = CenterHead(
            in_channels=64, heads=Config.HEADS, head_conv=Config.HEAD_CONV
        )

    def forward(self, batch_dict):
        """
        Args:
            batch_dict: {
                'points': List[Tensor],
                ...
            }
        """
        points = batch_dict["points"]

        # 1. Encode
        x = self.encoder(points)  # (B, 64, H, W)

        # 2. Backbone
        feats = self.backbone(x)  # List of features

        # 3. Neck / Upsample
        x = self.neck(feats)  # (B, 64, H/4, W/4)

        # 4. Head
        preds = self.head(x)

        # Apply sigmoid to heatmap and iou
        preds["hm"] = torch.sigmoid(preds["hm"])
        preds["iou"] = torch.sigmoid(preds["iou"])  # IoU is 0-1

        return preds
