import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import ModelConfig, VoxelConfig


class PillarFeatureNet(nn.Module):
    """
    Encodes raw point data within pillars into a feature vector.
    Architecture: Linear -> BN -> ReLU -> MaxPool
    """

    def __init__(self):
        super().__init__()
        self.in_channels = ModelConfig.num_input_features
        self.out_channels = ModelConfig.pfn_num_filters[0]

        # Linear layer to expand feature dimension
        self.linear = nn.Linear(self.in_channels, self.out_channels, bias=False)
        self.norm = nn.BatchNorm1d(self.out_channels)

    def forward(self, features):
        """
        Args:
            features: (M, max_points, num_input_features)
        Returns:
            pillar_features: (M, out_channels)
        """
        # features: (M, 32, 9)
        x = self.linear(features)  # (M, 32, 64)

        # Transpose for BatchNorm1d: (M, C, Points)
        x = x.permute(0, 2, 1)
        x = self.norm(x)
        x = F.relu(x)

        # Max pooling over the points dimension to get a single vector per pillar
        x_max = torch.max(x, dim=2)[0]  # (M, 64)
        return x_max


class PointPillarsScatter(nn.Module):
    """
    Scatters sparse pillar features into a dense 2D BEV pseudo-image.
    """

    def __init__(self):
        super().__init__()
        self.num_bev_features = ModelConfig.pfn_num_filters[0]

        # Instantiate VoxelConfig to access the property
        vc = VoxelConfig()
        self.grid_size = vc.grid_size  # [W, H, D]
        self.nx = self.grid_size[0]
        self.ny = self.grid_size[1]

    def forward(self, pillar_features, coords, batch_size):
        """
        Args:
            pillar_features: (M, C)
            coords: (M, 4) [batch_idx, z_idx, y_idx, x_idx]
            batch_size: int
        Returns:
            batch_canvas: (B, C, H, W)
        """
        # Create a zero-filled canvas
        canvas = torch.zeros(
            (batch_size, self.num_bev_features, self.ny, self.nx),
            dtype=pillar_features.dtype,
            device=pillar_features.device,
        )

        # Unpack coordinates
        batch_idx = coords[:, 0]
        y_idx = coords[:, 2]
        x_idx = coords[:, 3]

        # Create a mask to filter out invalid coordinates (sanity check)
        mask = (
            (batch_idx >= 0)
            & (batch_idx < batch_size)
            & (y_idx >= 0)
            & (y_idx < self.ny)
            & (x_idx >= 0)
            & (x_idx < self.nx)
        )

        # Apply mask
        batch_idx = batch_idx[mask]
        y_idx = y_idx[mask]
        x_idx = x_idx[mask]
        features = pillar_features[mask]

        # Scatter features onto the canvas
        # Note: If multiple pillars map to the same voxel, the last one overwrites.
        # In standard PointPillars, this is acceptable or handled by pre-filtering.
        canvas[batch_idx, :, y_idx, x_idx] = features

        return canvas


class BEVBackbone(nn.Module):
    """
    2D CNN Backbone for processing the BEV pseudo-image.
    Consists of a downsampling path and an upsampling path (FPN-like).
    """

    def __init__(self):
        super().__init__()

        input_channels = ModelConfig.backbone_input_channels
        layer_strides = ModelConfig.layer_strides
        layer_nums = ModelConfig.layer_nums
        num_filters = ModelConfig.num_filters
        upsample_strides = ModelConfig.upsample_strides
        num_upsample_filters = ModelConfig.num_upsample_filters

        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()

        c_in = input_channels

        # Build Downsampling Blocks
        for idx in range(len(layer_strides)):
            block_layers = []
            c_out = num_filters[idx]
            stride = layer_strides[idx]

            # First layer in the block handles the stride (downsampling)
            block_layers.append(
                nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False)
            )
            block_layers.append(nn.BatchNorm2d(c_out))
            block_layers.append(nn.ReLU(inplace=True))

            # Subsequent layers in the block maintain resolution
            for _ in range(layer_nums[idx] - 1):
                block_layers.append(nn.Conv2d(c_out, c_out, 3, padding=1, bias=False))
                block_layers.append(nn.BatchNorm2d(c_out))
                block_layers.append(nn.ReLU(inplace=True))

            self.blocks.append(nn.Sequential(*block_layers))

            # Build Upsampling Block for this level
            upsample_stride = upsample_strides[idx]
            c_up = num_upsample_filters[idx]

            if upsample_stride > 1:
                self.deblocks.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(
                            c_out,
                            c_up,
                            upsample_stride,
                            stride=upsample_stride,
                            bias=False,
                        ),
                        nn.BatchNorm2d(c_up),
                        nn.ReLU(inplace=True),
                    )
                )
            else:
                self.deblocks.append(
                    nn.Sequential(
                        nn.Conv2d(c_out, c_up, 3, padding=1, bias=False),
                        nn.BatchNorm2d(c_up),
                        nn.ReLU(inplace=True),
                    )
                )

            c_in = c_out

        self.out_channels = sum(num_upsample_filters)

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            x: (B, sum(upsample_filters), H, W)
        """
        ups = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            ups.append(self.deblocks[i](x))

        # Concatenate upsampled features from all scales
        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        else:
            x = ups[0]

        return x


class CenterHead(nn.Module):
    """
    Anchor-free detection head.
    Predicts heatmap and regression maps for 3D object attributes.
    """

    def __init__(self, in_channels):
        super().__init__()

        self.heads_config = ModelConfig.heads
        self.head_conv = ModelConfig.head_conv

        self.tasks = nn.ModuleDict()

        for head_name, num_output in self.heads_config.items():
            layers = []
            # Shared convolution for the head
            layers.append(
                nn.Conv2d(in_channels, self.head_conv, 3, padding=1, bias=True)
            )
            layers.append(nn.BatchNorm2d(self.head_conv))
            layers.append(nn.ReLU(inplace=True))

            # Final 1x1 convolution to project to output channels
            layers.append(nn.Conv2d(self.head_conv, num_output, 1, bias=True))

            self.tasks[head_name] = nn.Sequential(*layers)

        self.init_weights()

    def init_weights(self):
        for name, module in self.tasks.items():
            # Initialize the last layer (1x1 conv)
            last_layer = module[-1]
            if name == "hm":
                # Heatmap bias initialization: -2.19 corresponds to p=0.1
                # This prevents the initial loss from being dominated by the background
                nn.init.constant_(last_layer.bias, -2.19)
            else:
                nn.init.normal_(last_layer.weight, std=0.001)
                nn.init.constant_(last_layer.bias, 0)

    def forward(self, x):
        ret = {}
        for name, module in self.tasks.items():
            ret[name] = module(x)
        return ret


class CenterPointPillars(nn.Module):
    """
    End-to-End Pillar-based CenterPoint Detector.
    """

    def __init__(self):
        super().__init__()
        self.pfn = PillarFeatureNet()
        self.scatter = PointPillarsScatter()
        self.backbone = BEVBackbone()
        self.head = CenterHead(self.backbone.out_channels)

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: dict containing:
                - pillar_features: (Total_Pillars, Max_Points, Feat_Dim)
                - pillar_coords: (Total_Pillars, 4) [b, z, y, x]
                - batch_size: int
        Returns:
            preds_dict: dict of head outputs (hm, center_z, dim, rot, reg)
        """
        pillar_features = batched_inputs["pillar_features"]
        pillar_coords = batched_inputs["pillar_coords"]
        batch_size = batched_inputs["batch_size"]

        # 1. Pillar Feature Encoding
        # Output: (M, 64)
        features = self.pfn(pillar_features)

        # 2. Scatter to BEV
        # Output: (B, 64, H, W)
        spatial_features = self.scatter(features, pillar_coords, batch_size)

        # 3. Backbone
        # Output: (B, 384, H, W)
        bev_features = self.backbone(spatial_features)

        # 4. Detection Head
        # Output: Dict of tensors
        preds = self.head(bev_features)

        return preds
