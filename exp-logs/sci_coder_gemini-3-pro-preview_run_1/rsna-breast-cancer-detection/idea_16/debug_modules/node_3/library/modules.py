import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class DeformableAlignmentModule(nn.Module):
    """
    Module that aligns a contralateral feature map to a target feature map
    using predicted dense spatial offsets.
    """

    def __init__(self, channels):
        super().__init__()
        # Predict offsets: (dx, dy)
        # Input: concatenation of target and contra features (2 * channels)
        # We use a bottleneck structure or simple convs. Here simple convs for efficiency.
        self.offset_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 2, kernel_size=3, padding=1, bias=True),
        )

        # Initialize the last conv layer to 0.
        # This ensures the module starts with an identity transform (0 offsets).
        nn.init.constant_(self.offset_conv[-1].weight, 0)
        nn.init.constant_(self.offset_conv[-1].bias, 0)

    def forward(self, x_target, x_contra):
        """
        Args:
            x_target: Feature map from the target breast [B, C, H, W]
            x_contra: Feature map from the contralateral breast [B, C, H, W]
        Returns:
            x_aligned: Contralateral feature map warped to align with x_target
        """
        # 1. Predict Offsets
        # Concatenate features along channel dimension
        combined = torch.cat([x_target, x_contra], dim=1)
        offsets = self.offset_conv(combined)  # Output shape: [B, 2, H, W]

        # 2. Create Base Grid
        B, _, H, W = x_target.size()
        dtype = x_target.dtype
        device = x_target.device

        # Create meshgrid in normalized coordinates [-1, 1]
        # align_corners=True convention
        y_vals = torch.linspace(-1, 1, H, device=device, dtype=dtype)
        x_vals = torch.linspace(-1, 1, W, device=device, dtype=dtype)
        base_grid_y, base_grid_x = torch.meshgrid(y_vals, x_vals, indexing="ij")

        # Stack to [B, H, W, 2] -> (x, y)
        base_grid = (
            torch.stack([base_grid_x, base_grid_y], dim=-1)
            .unsqueeze(0)
            .expand(B, -1, -1, -1)
        )

        # 3. Apply Offsets
        # Permute offsets from [B, 2, H, W] to [B, H, W, 2] to match grid shape
        offsets = offsets.permute(0, 2, 3, 1)

        # Add offsets to base grid
        # Note: Offsets are learned in the normalized coordinate space [-1, 1]
        grid = base_grid + offsets

        # 4. Resample (Warp)
        # padding_mode='zeros' ensures that sampling outside the image returns 0
        # align_corners=True matches the grid generation
        x_aligned = F.grid_sample(
            x_contra, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )

        return x_aligned


class SiameseFPNModel(nn.Module):
    """
    FPN-Enhanced Deformable Siamese Network.
    Uses a shared EfficientNet backbone and FPN to extract features from both breasts.
    Aligns contralateral features to target features before computing the difference.
    """

    def __init__(self):
        super().__init__()

        # 1. Backbone
        # Create EfficientNet-B2 backbone
        # features_only=True returns intermediate feature maps
        # out_indices=(2, 3, 4) corresponds to strides 8, 16, 32 (P3, P4, P5)
        self.backbone = timm.create_model(
            Config.BACKBONE,
            pretrained=Config.PRETRAINED,
            features_only=True,
            out_indices=(2, 3, 4),
            in_chans=Config.NUM_CHANNELS,
        )

        # Get channel counts for the selected levels (P3, P4, P5)
        feature_channels = self.backbone.feature_info.channels()

        # 2. FPN Layers
        self.fpn_dim = 256

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.align_modules = nn.ModuleList()

        for in_c in feature_channels:
            # Lateral connection: 1x1 conv to reduce channels to fpn_dim
            self.lateral_convs.append(nn.Conv2d(in_c, self.fpn_dim, kernel_size=1))

            # Output convolution: 3x3 conv for feature smoothing
            self.fpn_convs.append(
                nn.Conv2d(self.fpn_dim, self.fpn_dim, kernel_size=3, padding=1)
            )

            # Alignment module for this level
            self.align_modules.append(DeformableAlignmentModule(self.fpn_dim))

        # 3. Classification Head
        # We concatenate GAP vectors from Target and Difference for all 3 levels.
        # Input dim = 3 (levels) * 2 (Target + Diff) * fpn_dim
        self.head_input_dim = len(feature_channels) * 2 * self.fpn_dim

        self.classifier = nn.Sequential(
            nn.Dropout(0.2), nn.Linear(self.head_input_dim, 1)
        )

    def _forward_fpn(self, x):
        """
        Extracts features and applies FPN.
        Returns list of FPN features [P3, P4, P5].
        """
        # Extract backbone features
        # features will be a list [feat_p3, feat_p4, feat_p5]
        features = self.backbone(x)

        # Apply lateral 1x1 convs
        laterals = [conv(f) for f, conv in zip(features, self.lateral_convs)]

        # Top-down pathway (Add upsampled higher-level features)
        # laterals indices: 0->P3, 1->P4, 2->P5

        # P5 -> P4
        laterals[1] = laterals[1] + F.interpolate(
            laterals[2], size=laterals[1].shape[-2:], mode="nearest"
        )

        # P4 -> P3
        laterals[0] = laterals[0] + F.interpolate(
            laterals[1], size=laterals[0].shape[-2:], mode="nearest"
        )

        # Apply smooth 3x3 convs
        fpn_features = [conv(lat) for lat, conv in zip(laterals, self.fpn_convs)]

        return fpn_features

    def forward(self, x_target, x_contra):
        """
        Args:
            x_target: [B, C, H, W] - Target breast image + metadata maps
            x_contra: [B, C, H, W] - Contralateral breast image + metadata maps
        Returns:
            logits: [B, 1] - Logits for cancer probability
        """
        # 1. Extract Features (Shared Weights)
        feats_target = self._forward_fpn(x_target)  # [P3, P4, P5]
        feats_contra = self._forward_fpn(x_contra)  # [P3, P4, P5]

        global_descriptors = []

        # 2. Process each FPN level
        for i in range(len(feats_target)):
            ft = feats_target[i]
            fc = feats_contra[i]

            # Align Contralateral to Target
            fc_aligned = self.align_modules[i](ft, fc)

            # Compute Difference (Asymmetry)
            # We subtract Aligned Contralateral from Target
            diff = ft - fc_aligned

            # Global Average Pooling
            # [B, C, H, W] -> [B, C]
            pool_target = torch.mean(ft, dim=(2, 3))
            pool_diff = torch.mean(diff, dim=(2, 3))

            global_descriptors.append(pool_target)
            global_descriptors.append(pool_diff)

        # 3. Concatenate and Classify
        # Flatten and concat all descriptors
        final_embedding = torch.cat(global_descriptors, dim=1)

        logits = self.classifier(final_embedding)

        return logits
