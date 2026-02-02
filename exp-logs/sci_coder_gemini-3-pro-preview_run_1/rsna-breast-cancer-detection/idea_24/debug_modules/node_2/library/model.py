import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class SiameseEfficientNet(nn.Module):
    """
    Shared backbone using EfficientNet-B2.
    Extracts features at levels P3 (stride 8), P4 (stride 16), and P5 (stride 32).
    """

    def __init__(
        self,
        backbone_name=Config.MODEL_BACKBONE,
        pretrained=True,
        in_chans=Config.IN_CHANNELS,
    ):
        super().__init__()
        # Load timm model with features_only=True to get intermediate layers
        # tf_efficientnet_b2_ns feature indices:
        # 0: stride 2, 1: stride 4, 2: stride 8, 3: stride 16, 4: stride 32
        self.encoder = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            in_chans=in_chans,
            out_indices=(2, 3, 4),
        )

        # Get channel counts for P3, P4, P5
        # For tf_efficientnet_b2_ns: P3=48, P4=120, P5=352
        feature_info = self.encoder.feature_info
        self.channels = [info["num_chs"] for info in feature_info]

    def forward(self, x):
        """
        Args:
            x: Input tensor (B, C, H, W)
        Returns:
            List of feature maps [P3, P4, P5]
        """
        return self.encoder(x)


class FeaturePyramid(nn.Module):
    """
    Feature Pyramid Network (FPN) to fuse multi-scale features.
    """

    def __init__(self, in_channels_list, out_channels=128):
        super().__init__()
        self.out_channels = out_channels

        # Lateral layers (1x1 conv) to reduce channels
        self.lateral_convs = nn.ModuleList(
            [
                nn.Conv2d(in_ch, out_channels, kernel_size=1)
                for in_ch in in_channels_list
            ]
        )

        # Output layers (3x3 conv) to reduce aliasing
        self.output_convs = nn.ModuleList(
            [
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
                for _ in in_channels_list
            ]
        )

    def forward(self, features):
        """
        Args:
            features: List of [P3, P4, P5] from backbone
        Returns:
            List of refined [P3, P4, P5] features with same channel count
        """
        # features[0] -> P3, features[1] -> P4, features[2] -> P5

        # 1. Lateral connections
        p5 = self.lateral_convs[2](features[2])
        p4 = self.lateral_convs[1](features[1])
        p3 = self.lateral_convs[0](features[0])

        # 2. Top-down pathway
        # Upsample P5 and add to P4
        p4 = p4 + F.interpolate(p5, size=p4.shape[-2:], mode="nearest")
        # Upsample P4 and add to P3
        p3 = p3 + F.interpolate(p4, size=p3.shape[-2:], mode="nearest")

        # 3. Smooth outputs
        p5_out = self.output_convs[2](p5)
        p4_out = self.output_convs[1](p4)
        p3_out = self.output_convs[0](p3)

        return [p3_out, p4_out, p5_out]


class AlignmentModule(nn.Module):
    """
    Predicts offsets and aligns the contralateral feature map to the target feature map
    using Deformable Alignment (Grid Sample).
    """

    def __init__(self, channels):
        super().__init__()
        # Predict flow field from concatenated features
        # Input: 2 * channels (Target + Contra)
        # Output: 2 channels (dx, dy)
        self.offset_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 2, kernel_size=3, padding=1, bias=False),
        )

        # Initialize weights to zero to start with identity mapping
        nn.init.constant_(self.offset_conv[-1].weight, 0)
        # nn.init.constant_(self.offset_conv[-1].bias, 0) # Bias is False

    def forward(self, target_feat, contra_feat):
        """
        Args:
            target_feat: (B, C, H, W)
            contra_feat: (B, C, H, W)
        Returns:
            aligned_contra_feat: (B, C, H, W)
        """
        B, C, H, W = target_feat.shape

        # 1. Predict offsets
        # Concatenate along channel dimension
        concat = torch.cat([target_feat, contra_feat], dim=1)
        offsets = self.offset_conv(concat)  # (B, 2, H, W)

        # 2. Create base identity grid
        # Meshgrid in range [-1, 1]
        y_base = torch.linspace(-1, 1, H, device=target_feat.device)
        x_base = torch.linspace(-1, 1, W, device=target_feat.device)
        grid_y, grid_x = torch.meshgrid(y_base, x_base, indexing="ij")
        base_grid = torch.stack([grid_x, grid_y], dim=0)  # (2, H, W)
        base_grid = base_grid.unsqueeze(0).repeat(B, 1, 1, 1)  # (B, 2, H, W)

        # 3. Add predicted offsets to base grid
        # Offsets output is usually small, we can scale it or apply tanh if needed.
        # Here we assume direct prediction in normalized coordinates.
        sampling_grid = base_grid + offsets

        # Permute to (B, H, W, 2) for grid_sample
        sampling_grid = sampling_grid.permute(0, 2, 3, 1)

        # 4. Warp contralateral features
        aligned_contra_feat = F.grid_sample(
            contra_feat,
            sampling_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )

        return aligned_contra_feat


class GatedFusionBlock(nn.Module):
    """
    Computes difference, generates attention mask, and gates the target features.
    """

    def __init__(self, channels):
        super().__init__()

        # Lightweight block to compute attention from difference
        self.attention_net = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, target_feat, aligned_contra_feat):
        """
        Args:
            target_feat: (B, C, H, W)
            aligned_contra_feat: (B, C, H, W)
        Returns:
            gated_feat: (B, C, H, W)
        """
        # 1. Compute Difference
        diff = target_feat - aligned_contra_feat

        # 2. Generate Attention Mask
        # Mask values close to 1 indicate asymmetry (region of interest)
        # Mask values close to 0 indicate symmetry (background/normal)
        mask = self.attention_net(diff)  # (B, 1, H, W)

        # 3. Gate Fusion
        # Multiply target features by mask.
        # This suppresses features where the difference is low (symmetric).
        gated_feat = target_feat * mask

        return gated_feat


class ClassifierHead(nn.Module):
    """
    Aggregates features from all levels and predicts cancer probability.
    """

    def __init__(self, in_channels, num_levels=3):
        super().__init__()
        total_channels = in_channels * num_levels

        self.head = nn.Sequential(
            nn.Linear(total_channels, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
        )

    def forward(self, features_list):
        """
        Args:
            features_list: List of gated feature maps [(B, C, H1, W1), (B, C, H2, W2), ...]
        Returns:
            logits: (B, 1)
        """
        pooled_feats = []
        for feat in features_list:
            # Global Average Pooling: (B, C, H, W) -> (B, C, 1, 1) -> (B, C)
            pooled = F.adaptive_avg_pool2d(feat, 1).flatten(1)
            pooled_feats.append(pooled)

        # Concatenate all levels
        cat_feats = torch.cat(pooled_feats, dim=1)

        # Predict
        logits = self.head(cat_feats)
        return logits


class AsymmetryGatedSiameseNetwork(nn.Module):
    """
    Main Model Class: Asymmetry-Gated Deformable Siamese EfficientNet-B2
    """

    def __init__(self):
        super().__init__()

        # 1. Shared Backbone
        self.backbone = SiameseEfficientNet()
        backbone_channels = self.backbone.channels  # [48, 120, 352] for B2

        # 2. Shared Feature Pyramid
        self.fpn_channels = 128
        self.fpn = FeaturePyramid(backbone_channels, out_channels=self.fpn_channels)

        # 3. Alignment and Gating Modules for each level
        # We have 3 levels (P3, P4, P5)
        self.align_modules = nn.ModuleList(
            [AlignmentModule(self.fpn_channels) for _ in range(3)]
        )
        self.gate_modules = nn.ModuleList(
            [GatedFusionBlock(self.fpn_channels) for _ in range(3)]
        )

        # 4. Classifier Head
        self.head = ClassifierHead(self.fpn_channels, num_levels=3)

    def forward(self, target_img, contra_img):
        """
        Args:
            target_img: (B, 3, H, W) - Image + Age + Implant
            contra_img: (B, 3, H, W) - Image + Age + Implant
        Returns:
            logits: (B, 1)
        """
        # 1. Feature Extraction (Shared Backbone)
        target_features = self.backbone(target_img)  # [P3, P4, P5]
        contra_features = self.backbone(contra_img)  # [P3, P4, P5]

        # 2. Feature Refinement (Shared FPN)
        target_fpn = self.fpn(target_features)  # [P3', P4', P5']
        contra_fpn = self.fpn(contra_features)  # [P3', P4', P5']

        gated_features_list = []

        # 3. Level-wise Alignment and Gating
        for i in range(3):
            t_feat = target_fpn[i]
            c_feat = contra_fpn[i]

            # Align Contralateral to Target
            c_aligned = self.align_modules[i](t_feat, c_feat)

            # Gated Fusion
            gated = self.gate_modules[i](t_feat, c_aligned)

            gated_features_list.append(gated)

        # 4. Classification
        logits = self.head(gated_features_list)

        return logits
