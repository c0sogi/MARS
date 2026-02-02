import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class DecoderBlock(nn.Module):
    """
    Standard U-Net Decoder Block using Bilinear Upsampling.
    Cite solution_lesson_node_00039: Prefer bilinear upsampling over PixelShuffle for segmentation of amorphous structures.
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

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
            if x.size(2) != skip.size(2) or x.size(3) != skip.size(3):
                x = F.interpolate(
                    x,
                    size=(skip.size(2), skip.size(3)),
                    mode="bilinear",
                    align_corners=True,
                )
            x = torch.cat([x, skip], dim=1)

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class ResNet18UNet(nn.Module):
    """
    ResNet18 U-Net with Standard Components.
    Cite solution_lesson_node_00008: Use Shallow Heads (AdaptiveAvgPool + Linear).
    """

    def __init__(self, num_classes=4, pretrained=True):
        super(ResNet18UNet, self).__init__()

        # --- Encoder (ResNet18) ---
        base = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )

        self.enc0 = nn.Sequential(base.conv1, base.bn1, base.relu)
        self.pool = base.maxpool
        self.enc1 = base.layer1
        self.enc2 = base.layer2
        self.enc3 = base.layer3
        self.enc4 = base.layer4

        # --- Classification Head (Study Level) ---
        # Cite solution_lesson_node_00035: Restrict global classification to bottleneck
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.cls_head = nn.Linear(512, num_classes)

        # --- Decoder (Image Level) ---
        self.dec4 = DecoderBlock(in_channels=512, skip_channels=256, out_channels=256)
        self.dec3 = DecoderBlock(in_channels=256, skip_channels=128, out_channels=128)
        self.dec2 = DecoderBlock(in_channels=128, skip_channels=64, out_channels=64)
        self.dec1 = DecoderBlock(in_channels=64, skip_channels=64, out_channels=32)

        # Final Upsample to restore full resolution
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        self.seg_head = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        # --- Encoder ---
        x0 = self.enc0(x)
        p0 = self.pool(x0)
        x1 = self.enc1(p0)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)

        # --- Classification Branch ---
        pooled = self.avgpool(x4)
        pooled = pooled.flatten(1)
        logits = self.cls_head(pooled)

        # --- Segmentation Branch ---
        d4 = self.dec4(x4, x3)
        d3 = self.dec3(d4, x2)
        d2 = self.dec2(d3, x1)
        d1 = self.dec1(d2, x0)

        out = self.final_up(d1)
        mask = self.seg_head(out)

        return mask, logits
