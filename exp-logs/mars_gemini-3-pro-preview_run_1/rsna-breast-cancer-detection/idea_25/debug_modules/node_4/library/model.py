import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library import config
from library.modules import DeformableAlignmentBlock, AsymmetryGatingBlock


class SiameseEfficientNet(nn.Module):
    """
    FPN-Enhanced Asymmetry-Gated Deformable Siamese EfficientNet-B2.

    This model processes a pair of images (Target and Contralateral) using a shared
    EfficientNet backbone and FPN. It aligns the contralateral features to the target
    features using deformable convolution, computes the difference, and uses an
    asymmetry gating mechanism to suppress symmetric background information in the
    target features before classification.
    """

    def __init__(self):
        super(SiameseEfficientNet, self).__init__()

        # 1. Shared Backbone
        # Initialize EfficientNet-B2 with pretrained weights
        # features_only=True returns intermediate feature maps
        self.backbone = timm.create_model(
            config.BACKBONE,
            pretrained=config.PRETRAINED,
            features_only=True,
            in_chans=config.IN_CHANNELS,
            drop_rate=config.DROP_RATE,
            drop_path_rate=config.DROP_PATH_RATE,
        )

        # Determine channel dimensions for P3, P4, P5 (indices 2, 3, 4)
        # We run a dummy forward pass to dynamically get shapes
        dummy_input = torch.zeros(1, config.IN_CHANNELS, 256, 256)
        with torch.no_grad():
            features = self.backbone(dummy_input)

        # EfficientNet features are usually at indices:
        # 0: stride 2
        # 1: stride 4
        # 2: stride 8  (P3)
        # 3: stride 16 (P4)
        # 4: stride 32 (P5)
        c3_channels = features[2].shape[1]
        c4_channels = features[3].shape[1]
        c5_channels = features[4].shape[1]

        fpn_channels = config.FPN_CHANNELS

        # 2. Shared FPN Layers
        # Lateral convolutions (1x1) to project backbone features to FPN dimension
        self.lat_layer3 = nn.Conv2d(c3_channels, fpn_channels, kernel_size=1)
        self.lat_layer4 = nn.Conv2d(c4_channels, fpn_channels, kernel_size=1)
        self.lat_layer5 = nn.Conv2d(c5_channels, fpn_channels, kernel_size=1)

        # Output convolutions (3x3) for feature smoothing
        self.fpn_layer3 = nn.Conv2d(
            fpn_channels, fpn_channels, kernel_size=3, padding=1
        )
        self.fpn_layer4 = nn.Conv2d(
            fpn_channels, fpn_channels, kernel_size=3, padding=1
        )
        self.fpn_layer5 = nn.Conv2d(
            fpn_channels, fpn_channels, kernel_size=3, padding=1
        )

        # 3. Alignment and Gating Modules
        # Create a block for each FPN level (P3, P4, P5)
        self.align_blocks = nn.ModuleList(
            [DeformableAlignmentBlock(fpn_channels) for _ in range(3)]
        )

        self.gate_blocks = nn.ModuleList(
            [AsymmetryGatingBlock(fpn_channels) for _ in range(3)]
        )

        # 4. Classification Head
        # Global Average Pooling applied to each level, then concatenated.
        # Input dim: FPN_CHANNELS * 3 (Levels)
        self.classifier = nn.Linear(fpn_channels * 3, 1)

    def _forward_fpn(self, x):
        """
        Passes input through backbone and FPN.
        Returns: List of feature maps [P3, P4, P5]
        """
        # Backbone forward
        features = self.backbone(x)
        c3, c4, c5 = features[2], features[3], features[4]

        # FPN Top-Down Pathway
        # P5
        p5 = self.lat_layer5(c5)

        # P4 = Lateral(C4) + Upsample(P5)
        p4 = self.lat_layer4(c4)
        p4 = p4 + F.interpolate(p5, size=p4.shape[-2:], mode="nearest")

        # P3 = Lateral(C3) + Upsample(P4)
        p3 = self.lat_layer3(c3)
        p3 = p3 + F.interpolate(p4, size=p3.shape[-2:], mode="nearest")

        # Smoothing
        p5 = self.fpn_layer5(p5)
        p4 = self.fpn_layer4(p4)
        p3 = self.fpn_layer3(p3)

        return [p3, p4, p5]

    def forward(self, x_target, x_contra):
        """
        Forward pass for the Siamese Network.

        Args:
            x_target (torch.Tensor): Target images (B, C, H, W)
            x_contra (torch.Tensor): Contralateral images (B, C, H, W)

        Returns:
            torch.Tensor: Logits (B, 1)
        """
        # 1. Extract Multi-scale Features (Shared Weights)
        # Returns [P3, P4, P5] for each branch
        feats_target = self._forward_fpn(x_target)
        feats_contra = self._forward_fpn(x_contra)

        gated_features_list = []

        # 2. Iterate through FPN levels
        for i in range(3):
            f_t = feats_target[i]
            f_c = feats_contra[i]

            # A. Deformable Alignment
            # Align contralateral features to match target geometry
            f_c_aligned = self.align_blocks[i](f_t, f_c)

            # B. Asymmetry Gating
            # Compute difference and gate target features
            f_t_gated = self.gate_blocks[i](f_t, f_c_aligned)

            # C. Global Average Pooling
            # (B, C, H, W) -> (B, C)
            gap = torch.mean(f_t_gated, dim=(2, 3))
            gated_features_list.append(gap)

        # 3. Classification Head
        # Concatenate features from all levels: (B, C*3)
        combined_features = torch.cat(gated_features_list, dim=1)

        # Predict logits
        logits = self.classifier(combined_features)

        return logits
