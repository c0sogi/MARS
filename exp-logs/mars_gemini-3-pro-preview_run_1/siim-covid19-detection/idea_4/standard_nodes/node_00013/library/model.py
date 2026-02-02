import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class DecoderBlock(nn.Module):
    """
    Standard U-Net Decoder Block: Upsample -> Concat -> ConvBlock.
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        # Input to conv is upsampled features + skip connection features
        self.conv1 = nn.Conv2d(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, skip=None):
        x = self.up(x)

        if skip is not None:
            # Handle potential rounding errors in dimensions during upsampling
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(
                    x, size=skip.shape[-2:], mode="bilinear", align_corners=True
                )
            x = torch.cat([x, skip], dim=1)

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class ResNet18Unet(nn.Module):
    """
    ResNet18 U-Net with simple linear classification head.
    Lightweight backbone chosen to prevent overfitting to coarse masks (Cite solution_lesson_node_00012).
    """

    def __init__(self):
        super(ResNet18Unet, self).__init__()

        # 1. Encoder (Backbone)
        # Load pre-trained ResNet18
        self.encoder = timm.create_model(
            Config.BACKBONE,
            pretrained=True,
            features_only=True,
            out_indices=(0, 1, 2, 3, 4),
        )

        # ResNet18 channels: [64, 64, 128, 256, 512]
        enc_channels = self.encoder.feature_info.channels()
        c0, c1, c2, c3, c4 = enc_channels

        # 2. Constrained Classification Head (Cite solution_lesson_node_00008)
        self.cls_head = nn.Linear(c4, Config.NUM_CLASSES)

        # 3. Decoder Path
        # Block 1: Up 32 (512) -> 16 (256)
        self.dec1 = DecoderBlock(c4, c3, 256)

        # Block 2: Up 16 (256) -> 8 (128)
        self.dec2 = DecoderBlock(256, c2, 128)

        # Block 3: Up 8 (128) -> 4 (64)
        self.dec3 = DecoderBlock(128, c1, 64)

        # Block 4: Up 4 (64) -> 2 (64)
        self.dec4 = DecoderBlock(64, c0, 32)

        # Block 5: Up 2 (32) -> 1 (Original)
        self.dec5 = DecoderBlock(32, 0, 16)

        # 4. Segmentation Head
        self.final_conv = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        # --- Encoder ---
        features = self.encoder(x)
        f0, f1, f2, f3, f4 = features

        # --- Classification Path ---
        x_cls = F.adaptive_avg_pool2d(f4, (1, 1)).flatten(1)
        study_logits = self.cls_head(x_cls)

        # --- Decoder Path ---
        d1 = self.dec1(f4, f3)
        d2 = self.dec2(d1, f2)
        d3 = self.dec3(d2, f1)
        d4 = self.dec4(d3, f0)
        d5 = self.dec5(d4)

        # --- Segmentation Head ---
        mask_logits = self.final_conv(d5)

        return study_logits, mask_logits
