import torch
import torch.nn as nn
import torch.nn.functional as F


class FeaturePyramidNetwork(nn.Module):
    """
    Shared Feature Pyramid Network (FPN) neck.
    Extracts and fuses features from backbone levels P3, P4, and P5.
    """

    def __init__(self, in_channels_list, out_channels=128):
        """
        Args:
            in_channels_list (list[int]): Channel counts for [P3, P4, P5].
                                          For EfficientNet-B2, typically [48, 120, 352].
            out_channels (int): The unified channel dimension for FPN outputs.
        """
        super().__init__()
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()

        for in_channels in in_channels_list:
            # 1x1 conv to reduce/project channels to out_channels
            self.inner_blocks.append(nn.Conv2d(in_channels, out_channels, 1))
            # 3x3 conv for output (reduces aliasing after upsampling)
            self.layer_blocks.append(
                nn.Conv2d(out_channels, out_channels, 3, padding=1)
            )

    def forward(self, features):
        """
        Args:
            features (list[torch.Tensor]): List of feature maps [P3, P4, P5].

        Returns:
            list[torch.Tensor]: List of FPN feature maps [P3_out, P4_out, P5_out].
        """
        # 1. Lateral connection for the top layer (P5)
        last_inner = self.inner_blocks[-1](features[-1])
        results = [self.layer_blocks[-1](last_inner)]

        # 2. Top-down pathway
        # Iterate from P4 down to P3
        for i in range(len(features) - 2, -1, -1):
            inner_lateral = self.inner_blocks[i](features[i])

            # Upsample the higher-level feature to match the current level's spatial size
            feat_shape = inner_lateral.shape[-2:]
            inner_top_down = F.interpolate(last_inner, size=feat_shape, mode="nearest")

            # Element-wise addition
            last_inner = inner_lateral + inner_top_down

            # Append processed result (insert at beginning to maintain P3, P4, P5 order)
            results.insert(0, self.layer_blocks[i](last_inner))

        return results


class DeformableAlignmentModule(nn.Module):
    """
    Predicts a dense flow field to align contralateral features to target features.
    Uses grid_sample for differentiable resampling.
    """

    def __init__(self, channels):
        """
        Args:
            channels (int): Number of channels in the input feature maps.
        """
        super().__init__()

        # Predicts flow (dx, dy) from concatenated features (2 * channels)
        self.offset_net = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(inplace=True),
            # Final layer predicts 2 channels: x_offset, y_offset
            nn.Conv2d(channels // 2, 2, kernel_size=3, padding=1, bias=True),
        )

        # Initialize final convolution weights/bias to 0
        # This ensures the module starts as an identity transform
        nn.init.constant_(self.offset_net[-1].weight, 0)
        nn.init.constant_(self.offset_net[-1].bias, 0)

    def forward(self, target_feat, contra_feat):
        """
        Args:
            target_feat (torch.Tensor): Feature map of the target breast (B, C, H, W).
            contra_feat (torch.Tensor): Feature map of the contralateral breast (B, C, H, W).

        Returns:
            torch.Tensor: Aligned contralateral feature map (B, C, H, W).
        """
        B, C, H, W = target_feat.shape

        # 1. Predict offsets
        concat = torch.cat([target_feat, contra_feat], dim=1)
        flow = self.offset_net(concat)  # Output shape: (B, 2, H, W)

        # 2. Create base normalized grid [-1, 1]
        # We create this on the fly to handle dynamic batch sizes/resolutions
        xx = torch.linspace(
            -1, 1, W, device=target_feat.device, dtype=target_feat.dtype
        )
        yy = torch.linspace(
            -1, 1, H, device=target_feat.device, dtype=target_feat.dtype
        )
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")

        base_grid = torch.stack([grid_x, grid_y], dim=0)  # (2, H, W)
        base_grid = base_grid.unsqueeze(0).repeat(B, 1, 1, 1)  # (B, 2, H, W)

        # 3. Add flow to grid
        # The flow is added directly to the normalized coordinates.
        # The network learns the appropriate scale.
        final_grid = base_grid + flow

        # Permute to (B, H, W, 2) required by grid_sample
        final_grid = final_grid.permute(0, 2, 3, 1)

        # 4. Resample
        # padding_mode='zeros' assumes regions outside the image are black (background)
        aligned_contra = F.grid_sample(
            contra_feat,
            final_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )

        return aligned_contra


class SpatialAttentionBlock(nn.Module):
    """
    Generates a spatial attention map from the difference feature map.
    Suppresses alignment artifacts and highlights asymmetric lesions.
    """

    def __init__(self, channels):
        """
        Args:
            channels (int): Number of channels in the difference map.
        """
        super().__init__()

        self.attn_net = nn.Sequential(
            # Reduce channels
            nn.Conv2d(channels, channels // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
            # Project to 1 channel spatial map
            nn.Conv2d(channels // 4, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Difference map (B, C, H, W).

        Returns:
            torch.Tensor: Weighted difference map (B, C, H, W).
        """
        attn_map = self.attn_net(x)  # (B, 1, H, W)
        return x * attn_map
