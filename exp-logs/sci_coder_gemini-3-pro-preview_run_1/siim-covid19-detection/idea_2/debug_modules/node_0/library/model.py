import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class DoubleConv(nn.Module):
    """
    (Conv2d -> BatchNorm -> ReLU) * 2
    Standard building block for U-Net.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class DecoderBlock(nn.Module):
    """
    Upscaling -> Concatenation -> DoubleConv
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        # We use bilinear interpolation for upsampling, so no learnable parameters here
        # The number of channels after concatenation will be in_channels + skip_channels
        self.conv = DoubleConv(in_channels + skip_channels, out_channels)

    def forward(self, x, skip=None):
        # Upsample x to match skip's spatial resolution
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)

        # If skip connection is provided, concatenate
        if skip is not None:
            # Handle potential padding issues if dimensions are not perfectly divisible
            if x.shape != skip.shape:
                x = F.interpolate(
                    x, size=skip.shape[2:], mode="bilinear", align_corners=True
                )
            x = torch.cat([x, skip], dim=1)

        return self.conv(x)


class EfficientNetUNet(nn.Module):
    def __init__(self, num_classes=Config.NUM_STUDY_CLASSES):
        super().__init__()

        # 1. Encoder (EfficientNet-B3)
        # features_only=True returns a list of feature maps
        self.encoder = timm.create_model(
            Config.BACKBONE,
            pretrained=True,
            features_only=True,
            out_indices=(0, 1, 2, 3, 4),  # Strides: 2, 4, 8, 16, 32
        )

        # Get channel counts from the encoder
        # For EfficientNet-B3: [24, 32, 48, 136, 384]
        enc_channels = self.encoder.feature_info.channels()

        # 2. Decoder
        # We build the decoder path going up from the bottleneck

        # Block 4: Input from Bottleneck (Stride 32), Skip from Stride 16
        # In: 384, Skip: 136 -> Out: 256
        self.dec4 = DecoderBlock(enc_channels[4], enc_channels[3], 256)

        # Block 3: Input from Dec4, Skip from Stride 8
        # In: 256, Skip: 48 -> Out: 128
        self.dec3 = DecoderBlock(256, enc_channels[2], 128)

        # Block 2: Input from Dec3, Skip from Stride 4
        # In: 128, Skip: 32 -> Out: 64
        self.dec2 = DecoderBlock(128, enc_channels[1], 64)

        # Block 1: Input from Dec2, Skip from Stride 2
        # In: 64, Skip: 24 -> Out: 32
        self.dec1 = DecoderBlock(64, enc_channels[0], 32)

        # Block 0: Final upsample to original resolution (Stride 1)
        # In: 32, Skip: None (or original image, but we skip that here) -> Out: 16
        self.dec0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        # 3. Heads

        # Segmentation Head
        # Maps 16 channels to 1 channel (binary mask logits)
        self.seg_head = nn.Conv2d(16, 1, kernel_size=1)

        # Classification Head
        # Global Average Pooling on the bottleneck features (Stride 32, 384 channels)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.cls_head = nn.Linear(enc_channels[4], num_classes)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input images (B, 3, H, W)
        Returns:
            seg_logits (torch.Tensor): Segmentation map logits (B, 1, H, W)
            cls_logits (torch.Tensor): Classification logits (B, 4)
        """
        # --- Encoder ---
        # features is a list of tensors: [f0, f1, f2, f3, f4]
        features = self.encoder(x)

        # Extract specific features for skip connections
        # f0: stride 2, f1: stride 4, f2: stride 8, f3: stride 16, f4: stride 32
        f0, f1, f2, f3, f4 = features

        # --- Classification Head ---
        # Use the deepest feature map (f4)
        cls_feat = self.global_pool(f4)
        cls_feat = torch.flatten(cls_feat, 1)
        cls_logits = self.cls_head(cls_feat)

        # --- Decoder ---
        x = self.dec4(f4, f3)
        x = self.dec3(x, f2)
        x = self.dec2(x, f1)
        x = self.dec1(x, f0)
        x = self.dec0(x)

        # --- Segmentation Head ---
        seg_logits = self.seg_head(x)

        return seg_logits, cls_logits
