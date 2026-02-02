import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class ResNetBlock(nn.Module):
    """
    Standard ResNet Block with 3x3 convolutions and a residual connection.
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class NarrowResNetEncoder(nn.Module):
    """
    Narrow ResNet Encoder with channel configuration [16, 32, 64].
    Designed for 32x32 input images, avoiding aggressive initial downsampling.
    """

    def __init__(self):
        super(NarrowResNetEncoder, self).__init__()
        channels = Config.ENCODER_CHANNELS  # Expected: [16, 32, 64]

        # Initial convolution: 3 -> 16. Stride 1 to preserve 32x32 resolution.
        self.init_conv = nn.Conv2d(
            3, channels[0], kernel_size=3, stride=1, padding=1, bias=False
        )
        self.init_bn = nn.BatchNorm2d(channels[0])
        self.init_relu = nn.ReLU(inplace=True)

        # Stage 1: 16 channels, 32x32 resolution
        self.layer1 = ResNetBlock(channels[0], channels[0], stride=1)

        # Stage 2: 32 channels, 16x16 resolution
        self.layer2 = ResNetBlock(channels[0], channels[1], stride=2)

        # Stage 3: 64 channels, 8x8 resolution (Bottleneck)
        self.layer3 = ResNetBlock(channels[1], channels[2], stride=2)

    def forward(self, x):
        # Initial processing
        x = self.init_relu(self.init_bn(self.init_conv(x)))

        # Encoder stages with skip connection extraction
        f1 = self.layer1(x)  # 32x32, 16ch
        f2 = self.layer2(f1)  # 16x16, 32ch
        f3 = self.layer3(f2)  # 8x8, 64ch (Bottleneck)

        return [f1, f2, f3]


class DecoderBlock(nn.Module):
    """
    Decoder block that upsamples features and merges with lateral skip connections.
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()

        # Convolutional block to process concatenated features
        # Input channels = upsampled channels (in_channels) + skip channels
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels + skip_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        # Upsample x to match skip spatial dimensions using bilinear interpolation
        x_up = F.interpolate(
            x, size=skip.shape[2:], mode="bilinear", align_corners=False
        )

        # Concatenate along channel dimension
        combined = torch.cat([x_up, skip], dim=1)

        # Refine features
        out = self.conv(combined)
        return out


class DeeplySupervisedUNet(nn.Module):
    """
    Custom Deeply Supervised Narrow ResNet-UNet.
    Features a dual-head architecture for deep supervision:
    1. Semantic Head on the bottleneck (high-level abstraction).
    2. Detail Head on the decoder output (fine-grained spatial recovery).
    """

    def __init__(self):
        super(DeeplySupervisedUNet, self).__init__()

        self.encoder = NarrowResNetEncoder()
        enc_ch = Config.ENCODER_CHANNELS  # [16, 32, 64]
        num_classes = Config.NUM_CLASSES

        # Decoder Pathway
        # Decoder 1: Upsample 8x8 (64ch) -> 16x16. Merge with 32ch skip. Output 32ch.
        self.decoder1 = DecoderBlock(
            in_channels=enc_ch[2], skip_channels=enc_ch[1], out_channels=enc_ch[1]
        )

        # Decoder 2: Upsample 16x16 (32ch) -> 32x32. Merge with 16ch skip. Output 16ch.
        self.decoder2 = DecoderBlock(
            in_channels=enc_ch[1], skip_channels=enc_ch[0], out_channels=enc_ch[0]
        )

        # Classification Heads
        # Semantic Head: Attached to Bottleneck (8x8, 64ch)
        self.semantic_head = nn.Linear(enc_ch[2], num_classes)

        # Detail Head: Attached to Final Decoder Output (32x32, 16ch)
        self.detail_head = nn.Linear(enc_ch[0], num_classes)

    def forward(self, x):
        # --- Encoder Pass ---
        # f1: 32x32, 16ch
        # f2: 16x16, 32ch
        # f3: 8x8, 64ch (Bottleneck)
        features = self.encoder(x)
        f1, f2, f3 = features[0], features[1], features[2]

        # --- Semantic Head (Deep Supervision) ---
        # Global Average Pooling on Bottleneck features
        semantic_feat = F.adaptive_avg_pool2d(f3, (1, 1)).flatten(1)
        semantic_logits = self.semantic_head(semantic_feat)

        # --- Decoder Pass ---
        # Recover spatial details
        d1 = self.decoder1(f3, f2)  # 16x16
        d2 = self.decoder2(d1, f1)  # 32x32

        # --- Detail Head (Final Output) ---
        # Global Average Pooling on recovered spatial features
        detail_feat = F.adaptive_avg_pool2d(d2, (1, 1)).flatten(1)
        detail_logits = self.detail_head(detail_feat)

        return semantic_logits, detail_logits
