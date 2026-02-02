import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_max
from library.config import (
    POINT_CLOUD_RANGE,
    VOXEL_SIZE,
    GRID_SIZE,
    NUM_POINT_FEATURES,
    NUM_FILTERS,
    BACKBONE_LAYERS,
    BACKBONE_CHANNELS,
    FPN_OUT_CHANNELS,
    COMMON_HEADS,
    HEAD_CONV,
    NUM_CLASSES,
)


class PillarVFE(nn.Module):
    def __init__(self, input_channels=4, output_channels=64):
        super().__init__()
        self.voxel_size = torch.tensor(VOXEL_SIZE).float()
        self.pc_range = torch.tensor(POINT_CLOUD_RANGE).float()
        self.grid_size = torch.tensor(GRID_SIZE).long()

        # Features: x, y, z, i, x-cx, y-cy, z-cz
        self.in_channels = input_channels + 3
        self.out_channels = output_channels

        self.linear = nn.Linear(self.in_channels, self.out_channels)
        self.norm = nn.BatchNorm1d(self.out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, batched_points):
        """
        Args:
            batched_points: (N, 5) tensor [batch_idx, x, y, z, i]
        Returns:
            batch_canvas: (B, C*Dz, H, W) dense pseudo-image
        """
        device = batched_points.device
        self.voxel_size = self.voxel_size.to(device)
        self.pc_range = self.pc_range.to(device)
        self.grid_size = self.grid_size.to(device)

        # 1. Calculate Grid Indices
        # points: (N, 4) -> x, y, z, i
        points = batched_points[:, 1:]
        batch_idx = batched_points[:, 0].long()

        # Coordinate relative to range start
        coords = points[:, :3] - self.pc_range[:3]

        # Quantize to grid indices
        voxel_inds = torch.div(coords, self.voxel_size, rounding_mode="floor").long()

        # Filter out of bounds
        # Check x, y, z limits
        valid_mask = (
            (voxel_inds[:, 0] >= 0)
            & (voxel_inds[:, 0] < self.grid_size[0])
            & (voxel_inds[:, 1] >= 0)
            & (voxel_inds[:, 1] < self.grid_size[1])
            & (voxel_inds[:, 2] >= 0)
            & (voxel_inds[:, 2] < self.grid_size[2])
        )

        points = points[valid_mask]
        batch_idx = batch_idx[valid_mask]
        voxel_inds = voxel_inds[valid_mask]
        coords = coords[valid_mask]  # Relative coords for valid points

        # 2. Augment Features with Offset from Voxel Center
        # Center = (index + 0.5) * size
        voxel_centers = (voxel_inds.float() + 0.5) * self.voxel_size
        center_offsets = coords - voxel_centers

        # Concatenate: [x, y, z, i, dx, dy, dz]
        features = torch.cat([points, center_offsets], dim=1)

        # 3. PointNet (Linear -> BN -> ReLU)
        x = self.linear(features)
        x = self.norm(x)
        x = self.relu(x)

        # 4. Scatter Max Pooling
        # We need a unique index for each voxel in the batch
        # Index = b * (Dz * Dy * Dx) + z * (Dy * Dx) + y * (Dx) + x
        # Note: GRID_SIZE is [W, H, D] or [X, Y, Z] based on config logic?
        # Config says: X=1280, Y=1280, Z=2.
        # voxel_inds order matches points: 0->x, 1->y, 2->z

        nx, ny, nz = self.grid_size[0], self.grid_size[1], self.grid_size[2]

        # Flattened index for scatter
        # Order: Batch, Z, Y, X
        flat_inds = (
            batch_idx * (nz * ny * nx)
            + voxel_inds[:, 2] * (ny * nx)
            + voxel_inds[:, 1] * (nx)
            + voxel_inds[:, 0]
        )

        # Max pool over points mapping to the same voxel
        # num_voxels = B * Z * Y * X
        batch_size = int(batched_points[:, 0].max().item()) + 1
        total_voxels = batch_size * nz * ny * nx

        # (Total_Voxels, C)
        voxel_features, _ = scatter_max(x, flat_inds, dim=0, dim_size=total_voxels)

        # 5. Reshape to Pseudo-Image
        # (B, Z, Y, X, C) -> (B, C, Z, Y, X)
        voxel_features = voxel_features.view(batch_size, nz, ny, nx, self.out_channels)
        voxel_features = voxel_features.permute(0, 4, 1, 2, 3).contiguous()

        # Collapse Z into Channels: (B, C*Z, Y, X)
        # Input to backbone is usually 2D.
        voxel_features = voxel_features.view(batch_size, self.out_channels * nz, ny, nx)

        return voxel_features


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetFPN(nn.Module):
    def __init__(self, input_channels, layers, channels, fpn_out_channels):
        super(ResNetFPN, self).__init__()
        self.in_planes = 64

        # Stem
        # Input is (B, 128, 1280, 1280)
        # We want to downsample quickly to manageable size.
        # Standard ResNet stem: 7x7 stride 2, MaxPool stride 2 -> Stride 4 total.
        self.conv1 = nn.Conv2d(
            input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Encoder Layers (ResNet-18 structure)
        # C2: Stride 4 (relative to input)
        self.layer1 = self._make_layer(BasicBlock, channels[0], layers[0], stride=1)
        # C3: Stride 8
        self.layer2 = self._make_layer(BasicBlock, channels[1], layers[1], stride=2)
        # C4: Stride 16
        self.layer3 = self._make_layer(BasicBlock, channels[2], layers[2], stride=2)
        # C5: Stride 32
        self.layer4 = self._make_layer(BasicBlock, channels[3], layers[3], stride=2)

        # FPN Layers
        self.lat_layer1 = nn.Conv2d(channels[0], fpn_out_channels, 1)
        self.lat_layer2 = nn.Conv2d(channels[1], fpn_out_channels, 1)
        self.lat_layer3 = nn.Conv2d(channels[2], fpn_out_channels, 1)
        self.lat_layer4 = nn.Conv2d(channels[3], fpn_out_channels, 1)

        self.smooth1 = nn.Conv2d(fpn_out_channels, fpn_out_channels, 3, padding=1)
        self.smooth2 = nn.Conv2d(fpn_out_channels, fpn_out_channels, 3, padding=1)
        self.smooth3 = nn.Conv2d(fpn_out_channels, fpn_out_channels, 3, padding=1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode="bilinear", align_corners=True) + y

    def forward(self, x):
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        c1 = self.maxpool(x)  # Stride 4

        # Encoder
        c2 = self.layer1(c1)  # Stride 4, 64 ch
        c3 = self.layer2(c2)  # Stride 8, 128 ch
        c4 = self.layer3(c3)  # Stride 16, 256 ch
        c5 = self.layer4(c4)  # Stride 32, 512 ch

        # FPN Top-Down
        p5 = self.lat_layer4(c5)
        p4 = self._upsample_add(p5, self.lat_layer3(c4))
        p3 = self._upsample_add(p4, self.lat_layer2(c3))
        p2 = self._upsample_add(p3, self.lat_layer1(c2))

        # Smoothing
        # We only need P2 for the high-res detection head
        p2 = self.smooth1(p2)

        return p2


class CenterHead(nn.Module):
    def __init__(self, in_channels, heads, head_conv):
        super(CenterHead, self).__init__()
        self.heads = heads
        self.in_channels = in_channels

        for head, num_c in heads.items():
            fc = nn.Sequential(
                nn.Conv2d(in_channels, head_conv, kernel_size=3, padding=1, bias=True),
                nn.BatchNorm2d(head_conv),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    head_conv, num_c, kernel_size=1, stride=1, padding=0, bias=True
                ),
            )

            # Initialization
            if "heatmap" in head:
                fc[-1].bias.data.fill_(-2.19)  # Focal loss init
            else:
                fc[-1].bias.data.fill_(0)

            self.__setattr__(head, fc)

    def forward(self, x):
        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(x)
        return ret


class PointPillarsResNetFPN(nn.Module):
    def __init__(self):
        super(PointPillarsResNetFPN, self).__init__()

        # 1. Voxel Feature Encoder
        # Input: 4 (x,y,z,i). Output: 64.
        # Grid Z=2, so VFE output to backbone is 64*2 = 128 channels.
        self.vfe = PillarVFE(
            input_channels=NUM_POINT_FEATURES, output_channels=NUM_FILTERS[0]
        )

        # 2. Backbone
        # Input channels: 128
        self.backbone = ResNetFPN(
            input_channels=NUM_FILTERS[0] * GRID_SIZE[2],
            layers=BACKBONE_LAYERS,
            channels=BACKBONE_CHANNELS,
            fpn_out_channels=FPN_OUT_CHANNELS,
        )

        # 3. Head
        # Add heatmap head to common heads
        heads = COMMON_HEADS.copy()
        heads["heatmap"] = NUM_CLASSES

        self.head = CenterHead(
            in_channels=FPN_OUT_CHANNELS, heads=heads, head_conv=HEAD_CONV
        )

    def forward(self, batched_points):
        # 1. VFE
        # (B, 128, H, W)
        x = self.vfe(batched_points)

        # 2. Backbone
        # (B, 256, H/4, W/4)
        x = self.backbone(x)

        # 3. Head
        preds = self.head(x)

        return preds
