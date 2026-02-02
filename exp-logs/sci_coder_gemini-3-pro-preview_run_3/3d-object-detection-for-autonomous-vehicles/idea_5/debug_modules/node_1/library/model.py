import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_max

from library.config import (
    POINT_CLOUD_RANGE,
    VOXEL_SIZE,
    GRID_SIZE,
    NUM_FILTERS,
    NUM_CLASSES,
)


class PillarFeatureNet(nn.Module):
    """
    Converts raw point cloud data into a dense 2D pseudo-image.
    Uses dynamic voxelization and scatter max-pooling for efficiency.
    """

    def __init__(self, num_input_features=4, num_filters=NUM_FILTERS):
        super().__init__()
        self.x_min = POINT_CLOUD_RANGE[0]
        self.y_min = POINT_CLOUD_RANGE[1]
        self.z_min = POINT_CLOUD_RANGE[2]
        self.x_max = POINT_CLOUD_RANGE[3]
        self.y_max = POINT_CLOUD_RANGE[4]
        self.z_max = POINT_CLOUD_RANGE[5]

        self.voxel_x = VOXEL_SIZE[0]
        self.voxel_y = VOXEL_SIZE[1]

        self.grid_x = GRID_SIZE[0]
        self.grid_y = GRID_SIZE[1]

        # Input features: x, y, z, i, dx, dy (offset from pillar center)
        self.in_channels = num_input_features + 2
        self.out_channels = num_filters

        self.linear = nn.Linear(self.in_channels, self.out_channels, bias=False)
        self.bn = nn.BatchNorm1d(self.out_channels)

    def forward(self, batch_points):
        """
        Args:
            batch_points: List of Tensors [(N1, 4), (N2, 4), ...]
        Returns:
            pseudo_image: (B, C, H, W)
        """
        batch_size = len(batch_points)
        device = batch_points[0].device

        # Prepare storage
        all_points = []
        batch_indices = []

        for b_idx, points in enumerate(batch_points):
            # Filter points outside range
            mask = (
                (points[:, 0] >= self.x_min)
                & (points[:, 0] < self.x_max)
                & (points[:, 1] >= self.y_min)
                & (points[:, 1] < self.y_max)
                & (points[:, 2] >= self.z_min)
                & (points[:, 2] < self.z_max)
            )
            points = points[mask]

            if points.shape[0] == 0:
                continue

            # Calculate grid indices
            coor_x = ((points[:, 0] - self.x_min) / self.voxel_x).long()
            coor_y = ((points[:, 1] - self.y_min) / self.voxel_y).long()

            # Calculate offsets from pillar center
            center_x = coor_x.float() * self.voxel_x + self.x_min + self.voxel_x / 2.0
            center_y = coor_y.float() * self.voxel_y + self.y_min + self.voxel_y / 2.0

            dx = points[:, 0] - center_x
            dy = points[:, 1] - center_y

            # Augment features
            features = torch.cat([points, dx.unsqueeze(1), dy.unsqueeze(1)], dim=1)

            all_points.append(features)

            # Calculate flat index for scatter
            # Index = batch_idx * H * W + y * W + x
            flat_idx = b_idx * self.grid_y * self.grid_x + coor_y * self.grid_x + coor_x
            batch_indices.append(flat_idx)

        if not all_points:
            return torch.zeros(
                (batch_size, self.out_channels, self.grid_y, self.grid_x),
                device=device,
            )

        all_points = torch.cat(all_points, dim=0)  # (Total_N, 6)
        batch_indices = torch.cat(batch_indices, dim=0)  # (Total_N)

        # MLP
        x = self.linear(all_points)
        x = self.bn(x)
        x = F.relu(x)

        # Max Pooling via Scatter
        # Output size: B * H * W
        total_voxels = batch_size * self.grid_y * self.grid_x
        pooled, _ = scatter_max(x, batch_indices, dim=0, dim_size=total_voxels)

        # Reshape to (B, H, W, C) -> (B, C, H, W)
        pseudo_image = pooled.view(
            batch_size, self.grid_y, self.grid_x, self.out_channels
        )
        pseudo_image = pseudo_image.permute(0, 3, 1, 2).contiguous()

        return pseudo_image


class Backbone(nn.Module):
    """
    2D CNN Backbone to extract spatial features from the pseudo-image.
    Downsamples the input by a factor of 4 (1024 -> 256).
    """

    def __init__(self, in_channels, out_channels=256):
        super().__init__()

        # Layer 1: Stride 1 (1024x1024)
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Layer 2: Stride 2 (512x512)
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # Layer 3: Stride 2 (256x256)
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class CenterHead(nn.Module):
    """
    Multi-head detection layer.
    Outputs: Heatmap, Dimensions, Rotation, Regression, Z-coordinate.
    """

    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Task Heads
        self.heatmap = nn.Conv2d(64, num_classes, kernel_size=1, bias=True)
        self.dim = nn.Conv2d(64, 3, kernel_size=1, bias=True)  # log(l, w, h)
        self.rot = nn.Conv2d(64, 2, kernel_size=1, bias=True)  # sin, cos
        self.reg = nn.Conv2d(64, 2, kernel_size=1, bias=True)  # dx, dy
        self.z_map = nn.Conv2d(64, 1, kernel_size=1, bias=True)  # z

        self._init_weights()

    def _init_weights(self):
        # Initialize heatmap bias to -2.19 (approx log(1/99)) for Focal Loss stability
        self.heatmap.bias.data.fill_(-2.19)

        # Initialize other heads
        for m in [self.dim, self.rot, self.reg, self.z_map]:
            nn.init.normal_(m.weight, std=0.001)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.shared_conv(x)

        hm = self.heatmap(x)
        dim = self.dim(x)
        rot = self.rot(x)
        reg = self.reg(x)
        z = self.z_map(x)

        return hm, dim, rot, reg, z


class CenterPointNet(nn.Module):
    """
    End-to-End Pillar-based CenterNet for 3D Object Detection.
    """

    def __init__(self):
        super().__init__()
        self.vfe = PillarFeatureNet(num_input_features=4, num_filters=NUM_FILTERS)
        self.backbone = Backbone(in_channels=NUM_FILTERS, out_channels=256)
        self.head = CenterHead(in_channels=256, num_classes=NUM_CLASSES)

    def forward(self, batch_dict):
        points = batch_dict["points"]  # List of tensors

        # 1. Voxelization & Feature Encoding
        x = self.vfe(points)  # (B, 64, 1024, 1024)

        # 2. Backbone
        x = self.backbone(x)  # (B, 256, 256, 256)

        # 3. Head
        hm, dim, rot, reg, z = self.head(x)

        return {"heatmap": hm, "dim": dim, "rot": rot, "reg": reg, "z_map": z}
