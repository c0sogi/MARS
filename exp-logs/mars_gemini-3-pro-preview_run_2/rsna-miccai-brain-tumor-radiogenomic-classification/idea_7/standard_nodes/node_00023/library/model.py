import torch
import torch.nn as nn
import timm


class ModalityGatedEfficientNet(nn.Module):
    """
    Modality-Gated EfficientNet-B0.

    Architecture:
    1. Grouped Convolutional Stem (groups=4) to isolate 4 modalities (12 channels total).
    2. Robust Weight Initialization (replicating pre-trained RGB edge detectors).
    3. EfficientNet-B0 Backbone.
    4. Regularized Classification Head (Dropout + Linear).
    """

    def __init__(self, num_classes=1, pretrained=True, dropout_rate=0.2):
        super(ModalityGatedEfficientNet, self).__init__()

        # Load EfficientNet-B0 backbone
        # We load with num_classes=0 to get the features, but we will access internals manually
        self.backbone = timm.create_model(
            "efficientnet_b0", pretrained=pretrained, num_classes=0
        )

        # ----------------------------------------------------------------------
        # 1. Modify Stem for 12-Channel Input & Modality Isolation
        # ----------------------------------------------------------------------
        old_conv = self.backbone.conv_stem

        # Configuration
        out_channels = old_conv.out_channels  # 32 for B0
        kernel_size = old_conv.kernel_size
        stride = old_conv.stride
        padding = old_conv.padding
        groups = 4  # Isolate the 4 modalities

        # Create new stem convolution
        self.stem_conv = nn.Conv2d(
            in_channels=12,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )

        # ----------------------------------------------------------------------
        # 2. Robust Initialization (Replication Strategy)
        # ----------------------------------------------------------------------
        if pretrained:
            # Original weights shape: (32, 3, 3, 3)
            # With groups=4, PyTorch expects weights of shape (Out, In/Groups, K, K)
            # which is (32, 12/4, 3, 3) -> (32, 3, 3, 3). The shape is identical.

            old_weights = old_conv.weight.data

            # Strategy: To ensure every modality gets high-quality edge detectors
            # (and not random texture filters from deeper in the bank), we take the
            # first 'filters_per_group' filters and replicate them for each group.
            filters_per_group = out_channels // groups  # 32 // 4 = 8

            # Select first 8 filters (usually primary edge detectors in CNNs)
            base_weights = old_weights[:filters_per_group, :, :, :]  # (8, 3, 3, 3)

            # Replicate 4 times along the output channel dimension
            # Result shape: (32, 3, 3, 3)
            new_weights = torch.cat([base_weights] * groups, dim=0)

            self.stem_conv.weight.data = new_weights

        # Keep original BN and Act
        self.stem_bn = self.backbone.bn1
        # Robustly get activation (timm 1.0+ might not have act1, B0 uses SiLU)
        self.stem_act = getattr(
            self.backbone, "act1", getattr(self.backbone, "act", nn.SiLU(inplace=True))
        )

        # ----------------------------------------------------------------------
        # 3. Regularized Head
        # ----------------------------------------------------------------------
        self.num_features = self.backbone.num_features

        # Robustly get head activation (act2 might be missing/renamed in newer timm)
        # Cite debug_lesson_4: Audit All Object Interactions.
        self.head_act = getattr(
            self.backbone, "act2", getattr(self.backbone, "act", nn.SiLU(inplace=True))
        )

        # Custom head with Dropout
        self.head = nn.Sequential(
            nn.Dropout(p=dropout_rate), nn.Linear(self.num_features, num_classes)
        )

    def forward(self, x):
        # x shape: (B, 12, H, W)

        # --- Custom Stem Forward ---
        x = self.stem_conv(x)
        x = self.stem_bn(x)
        x = self.stem_act(x)
        # Cite solution_lesson_node_00022: Removed early SE block to prevent suppression of localized signals

        # --- Backbone Forward ---
        # Pass through MBConv blocks
        x = self.backbone.blocks(x)

        # Pass through Backbone Head (Conv + BN + Act)
        x = self.backbone.conv_head(x)
        x = self.backbone.bn2(x)
        x = self.head_act(x)

        # --- Pooling ---
        x = self.backbone.global_pool(x)

        # --- Classification Head ---
        # Flatten if necessary (timm global_pool usually returns (B, C) or (B, C, 1, 1))
        if x.dim() == 4:
            x = x.flatten(1)

        x = self.head(x)

        return x
