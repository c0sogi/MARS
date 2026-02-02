import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class FPNNeck(nn.Module):
    """
    Feature Pyramid Network (FPN) Neck.

    Constructs a top-down pathway with lateral connections to build high-level
    semantic feature maps at all scales.
    """

    def __init__(self, in_channels_list, out_channels=128):
        super(FPNNeck, self).__init__()
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        # in_channels_list corresponds to features from bottom to top [C2, C3, C4, C5]
        # We build the pyramid from top to bottom (P5 -> P2)

        for in_c in in_channels_list:
            self.lateral_convs.append(nn.Conv2d(in_c, out_channels, kernel_size=1))
            self.fpn_convs.append(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            )

    def forward(self, inputs):
        """
        Args:
            inputs (list): List of feature maps [C2, C3, C4, C5] from the encoder.

        Returns:
            list: List of FPN feature maps [P2, P3, P4, P5].
        """
        # Build lateral projections
        laterals = [conv(x) for conv, x in zip(self.lateral_convs, inputs)]

        # Build top-down pathway
        # laterals is [L2, L3, L4, L5]
        # P5 = L5
        # P4 = L4 + Upsample(P5)
        # ...

        used_laterals = list(laterals)
        num_levels = len(used_laterals)

        # Iterate backwards from the second-to-last level down to the first
        for i in range(num_levels - 2, -1, -1):
            # Upsample the higher level feature map to match the current level's spatial size
            prev_shape = used_laterals[i].shape[2:]
            upsampled = F.interpolate(
                used_laterals[i + 1], size=prev_shape, mode="nearest"
            )
            used_laterals[i] = used_laterals[i] + upsampled

        # Apply smoothing convolutions (3x3) to generate final P-levels
        outs = [self.fpn_convs[i](used_laterals[i]) for i in range(num_levels)]

        return outs


class SegmentationHead(nn.Module):
    """
    Segmentation Head for FPN.

    Aggregates multi-scale features by upsampling them to a common resolution (1/4 scale),
    concatenating them, and applying final classification convolutions.
    """

    def __init__(self, in_channels, num_classes, scale_factors):
        super(SegmentationHead, self).__init__()
        self.scale_factors = scale_factors

        # Calculate total channels after concatenation
        # We have 4 levels (P2, P3, P4, P5), each with `in_channels` depth
        concat_channels = in_channels * len(scale_factors)

        self.conv_seg = nn.Sequential(
            nn.Conv2d(concat_channels, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1),
        )

    def forward(self, inputs):
        """
        Args:
            inputs (list): List of FPN feature maps [P2, P3, P4, P5].

        Returns:
            torch.Tensor: Raw logits at 1/4 input resolution.
        """
        upsampled_inputs = []
        # Target size is the spatial size of P2 (the largest feature map, 1/4 scale)
        target_size = inputs[0].shape[2:]

        for x, scale in zip(inputs, self.scale_factors):
            if scale > 1:
                x = F.interpolate(
                    x, size=target_size, mode="bilinear", align_corners=False
                )
            upsampled_inputs.append(x)

        # Concatenate all levels
        x = torch.cat(upsampled_inputs, dim=1)

        # Final prediction
        x = self.conv_seg(x)
        return x


class EfficientNetFPN(nn.Module):
    """
    2.5D EfficientNet-FPN Segmentation Model.

    Encoder: EfficientNet-B0 (strides 4, 8, 16, 32)
    Neck: FPN (Top-down pathway + Lateral connections)
    Head: Semantic FPN Head (Upsample + Concat)
    """

    def __init__(
        self,
        encoder_name=Config.MODEL_NAME,
        num_classes=Config.NUM_CLASSES,
        pretrained=True,
    ):
        super(EfficientNetFPN, self).__init__()

        # 1. Encoder
        # features_only=True returns a list of feature maps
        # out_indices=(1, 2, 3, 4) corresponds to strides 4, 8, 16, 32 for EfficientNet
        self.encoder = timm.create_model(
            encoder_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(1, 2, 3, 4),
            in_chans=Config.IN_CHANNELS,
        )

        # Determine encoder channel counts dynamically
        # Create a dummy input to trace shapes
        with torch.no_grad():
            dummy_input = torch.randn(1, Config.IN_CHANNELS, 256, 256)
            features = self.encoder(dummy_input)
            encoder_channels = [f.shape[1] for f in features]
            # Typically [24, 40, 112, 320] for EfficientNet-B0

        # 2. Neck
        # We use a fixed dimension of 128 for the FPN layers to keep it lightweight
        self.fpn_dim = 128
        self.neck = FPNNeck(encoder_channels, out_channels=self.fpn_dim)

        # 3. Head
        # P2 is 1/4 scale. P3(1/8), P4(1/16), P5(1/32)
        # Relative upsampling factors to match P2: 1, 2, 4, 8
        self.head = SegmentationHead(
            in_channels=self.fpn_dim,
            num_classes=num_classes,
            scale_factors=[1, 2, 4, 8],
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, 3, H, W).

        Returns:
            torch.Tensor: Logits of shape (B, num_classes, H, W).
        """
        input_shape = x.shape[2:]

        # Encoder pass
        features = self.encoder(x)  # Returns [C2, C3, C4, C5]

        # Neck pass
        pyramid_features = self.neck(features)  # Returns [P2, P3, P4, P5]

        # Head pass (outputs at 1/4 scale)
        logits = self.head(pyramid_features)

        # Final Upsample to original input resolution (4x)
        logits = F.interpolate(
            logits, size=input_shape, mode="bilinear", align_corners=False
        )

        return logits
