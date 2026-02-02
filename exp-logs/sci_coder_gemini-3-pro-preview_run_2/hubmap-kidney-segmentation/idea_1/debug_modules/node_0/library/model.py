import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from library.config import Config


class FPNResNet18(nn.Module):
    """
    Feature Pyramid Network (FPN) with ResNet-18 Encoder for FTU Detection.

    Architecture:
    - Backbone: ResNet-18 (pretrained on ImageNet)
    - Neck: FPN with lateral connections and top-down pathway
    - Head: Segmentation head that coalesces multi-scale features and upsamples to full resolution
    """

    def __init__(self, num_classes=Config.CLASSES, fpn_channels=128):
        """
        Args:
            num_classes (int): Number of output classes (1 for binary segmentation).
            fpn_channels (int): Number of channels in the FPN layers.
        """
        super(FPNResNet18, self).__init__()

        # 1. Encoder: ResNet-18
        # Using the modern weights API
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        backbone = torchvision.models.resnet18(weights=weights)

        # Extract specific layers from the backbone
        # Initial stem: conv1 (stride 2) -> bn1 -> relu -> maxpool (stride 2) => Total stride 4
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        # ResNet Stages
        self.layer1 = backbone.layer1  # C2: 64 channels, stride 4
        self.layer2 = backbone.layer2  # C3: 128 channels, stride 8
        self.layer3 = backbone.layer3  # C4: 256 channels, stride 16
        self.layer4 = backbone.layer4  # C5: 512 channels, stride 32

        # 2. FPN Lateral Connections (1x1 Convs)
        # Project all encoder levels to fpn_channels
        self.lat_layer1 = nn.Conv2d(64, fpn_channels, kernel_size=1)
        self.lat_layer2 = nn.Conv2d(128, fpn_channels, kernel_size=1)
        self.lat_layer3 = nn.Conv2d(256, fpn_channels, kernel_size=1)
        self.lat_layer4 = nn.Conv2d(512, fpn_channels, kernel_size=1)

        # 3. FPN Smoothing Layers (3x3 Convs)
        # Applied after adding the top-down pathway
        self.smooth_layer1 = nn.Conv2d(
            fpn_channels, fpn_channels, kernel_size=3, padding=1
        )
        self.smooth_layer2 = nn.Conv2d(
            fpn_channels, fpn_channels, kernel_size=3, padding=1
        )
        self.smooth_layer3 = nn.Conv2d(
            fpn_channels, fpn_channels, kernel_size=3, padding=1
        )
        self.smooth_layer4 = nn.Conv2d(
            fpn_channels, fpn_channels, kernel_size=3, padding=1
        )

        # 4. Segmentation Head
        # Coalesce features: Concatenate P2, P3, P4, P5 (all upsampled to P2 size)
        # Input channels = fpn_channels * 4
        self.coalesce_conv = nn.Sequential(
            nn.Conv2d(
                fpn_channels * 4, fpn_channels, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(fpn_channels),
            nn.ReLU(inplace=True),
        )

        # Upsample block to get from P2 (1/4 scale) to Original (1/1 scale)
        self.upsample_block = nn.Sequential(
            # Upsample 2x (1/4 -> 1/2)
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(fpn_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Upsample 2x (1/2 -> 1/1)
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Final prediction layer
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        # --- Encoder ---
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        c2 = self.layer1(x)  # Stride 4
        c3 = self.layer2(c2)  # Stride 8
        c4 = self.layer3(c3)  # Stride 16
        c5 = self.layer4(c4)  # Stride 32

        # --- FPN Top-Down Pathway ---
        # P5
        p5 = self.lat_layer4(c5)

        # P4: Add upsampled P5 to lateral C4
        p5_up = F.interpolate(p5, size=c4.shape[2:], mode="nearest")
        p4 = self.lat_layer3(c4) + p5_up

        # P3: Add upsampled P4 to lateral C3
        p4_up = F.interpolate(p4, size=c3.shape[2:], mode="nearest")
        p3 = self.lat_layer2(c3) + p4_up

        # P2: Add upsampled P3 to lateral C2
        p3_up = F.interpolate(p3, size=c2.shape[2:], mode="nearest")
        p2 = self.lat_layer1(c2) + p3_up

        # --- Smoothing ---
        p5 = self.smooth_layer4(p5)
        p4 = self.smooth_layer3(p4)
        p3 = self.smooth_layer2(p3)
        p2 = self.smooth_layer1(p2)

        # --- Segmentation Head (Coalescing) ---
        # Upsample all levels to the resolution of P2 (Stride 4)
        target_size = p2.shape[2:]

        p5_u = F.interpolate(p5, size=target_size, mode="bilinear", align_corners=False)
        p4_u = F.interpolate(p4, size=target_size, mode="bilinear", align_corners=False)
        p3_u = F.interpolate(p3, size=target_size, mode="bilinear", align_corners=False)
        # p2 is already at target_size

        # Concatenate features
        cat_features = torch.cat([p2, p3_u, p4_u, p5_u], dim=1)

        # Fuse features
        fused = self.coalesce_conv(cat_features)

        # Upsample to full resolution
        out = self.upsample_block(fused)

        # Final Logits
        logits = self.final_conv(out)

        return logits
