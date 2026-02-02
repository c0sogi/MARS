import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels + skip_channels, out_channels, 3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(
                    x, size=skip.shape[-2:], mode="bilinear", align_corners=False
                )
            x = torch.cat([x, skip], dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class CatheterModel(nn.Module):
    """
    Multi-Task Learning Model for Catheter Detection.
    Uses EfficientNetV2-S backbone with a U-Net style decoder for auxiliary segmentation.
    Cite solution_lesson_node_00020 (Dense Decoder Superiority)
    """

    def __init__(self, pretrained=True):
        super(CatheterModel, self).__init__()

        # 1. Backbone: EfficientNetV2-S
        # Extract features at indices 0, 1, 2, 3, 4
        # Stride 2, 4, 8, 16, 32
        self.backbone = timm.create_model(
            Config.BACKBONE,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3, 4),
        )

        # Dummy input to get channel counts
        dummy = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            feats = self.backbone(dummy)
        ch = [f.shape[1] for f in feats]
        # ch: [s2, s4, s8, s16, s32]

        # 2. Classification Head
        # Global Avg Pooling on the last feature map (Stride 32)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(Config.DROP_RATE)
        self.fc = nn.Linear(ch[-1], Config.NUM_CLASSES)

        # 3. Auxiliary Segmentation Decoder (U-Net Style)
        # Center: Stride 32 -> 256 channels
        # Dec4: In(256), Skip(s16), Out(128)
        self.dec4 = DecoderBlock(ch[4], ch[3], 128)
        # Dec3: In(128), Skip(s8), Out(64)
        self.dec3 = DecoderBlock(128, ch[2], 64)
        # Dec2: In(64), Skip(s4), Out(32)
        self.dec2 = DecoderBlock(64, ch[1], 32)
        # Dec1: In(32), Skip(s2), Out(16)
        self.dec1 = DecoderBlock(32, ch[0], 16)

        # Final Segmentation Head
        self.final_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(16, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
        )

    def forward(self, x):
        input_size = x.shape[-2:]

        # Encoder
        # f0: s2, f1: s4, f2: s8, f3: s16, f4: s32
        feats = self.backbone(x)

        # Classification Path
        x_cls = self.global_pool(feats[-1])
        x_cls = x_cls.flatten(1)
        x_cls = self.drop(x_cls)
        logits = self.fc(x_cls)

        # Segmentation Path (Decoder)
        # Start from f4 (s32)
        d4 = self.dec4(feats[4], feats[3])  # -> s16
        d3 = self.dec3(d4, feats[2])  # -> s8
        d2 = self.dec2(d3, feats[1])  # -> s4
        d1 = self.dec1(d2, feats[0])  # -> s2

        # Final upsample to s1
        mask = self.final_conv(d1)

        # Ensure mask matches input size exactly (in case of odd dims)
        if mask.shape[-2:] != input_size:
            mask = F.interpolate(
                mask, size=input_size, mode="bilinear", align_corners=False
            )

        return logits, mask
