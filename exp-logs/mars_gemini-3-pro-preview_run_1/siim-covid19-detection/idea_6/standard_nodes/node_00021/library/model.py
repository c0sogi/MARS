import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights
from library.config import Config


class ResNet34FPN(nn.Module):
    """
    ResNet34 with Feature Pyramid Network (FPN).

    Architecture:
    - Backbone: Pretrained ResNet34
    - Neck: FPN (P2, P3, P4, P5) with 256 channels
    - Head 1 (Classification): Multi-scale GAP (P3+P4+P5) -> Linear -> 4 Classes
    - Head 2 (Segmentation): Aggregated FPN features (P2..P5) -> Conv -> 1 Class (Opacity)
    """

    def __init__(self):
        super(ResNet34FPN, self).__init__()

        # =====================================================================
        # 1. Backbone: ResNet34
        # =====================================================================
        # Use modern torchvision weights API
        weights = ResNet34_Weights.DEFAULT
        backbone = resnet34(weights=weights)

        # Extract Stem (Input -> Stride 4)
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        # Extract Encoder Stages
        # layer1: 64 channels, Stride 4 (relative to input image)
        self.layer1 = backbone.layer1
        # layer2: 128 channels, Stride 8
        self.layer2 = backbone.layer2
        # layer3: 256 channels, Stride 16
        self.layer3 = backbone.layer3
        # layer4: 512 channels, Stride 32
        self.layer4 = backbone.layer4

        # =====================================================================
        # 2. FPN Neck
        # =====================================================================
        self.fpn_dim = 256

        # Lateral Connections (1x1 convs to project backbone channels to FPN dim)
        self.lat_layer1 = nn.Conv2d(64, self.fpn_dim, kernel_size=1)
        self.lat_layer2 = nn.Conv2d(128, self.fpn_dim, kernel_size=1)
        self.lat_layer3 = nn.Conv2d(256, self.fpn_dim, kernel_size=1)
        self.lat_layer4 = nn.Conv2d(512, self.fpn_dim, kernel_size=1)

        # Smoothing Convolutions (3x3 convs to reduce aliasing)
        self.smooth_layer1 = nn.Conv2d(
            self.fpn_dim, self.fpn_dim, kernel_size=3, padding=1
        )
        self.smooth_layer2 = nn.Conv2d(
            self.fpn_dim, self.fpn_dim, kernel_size=3, padding=1
        )
        self.smooth_layer3 = nn.Conv2d(
            self.fpn_dim, self.fpn_dim, kernel_size=3, padding=1
        )
        self.smooth_layer4 = nn.Conv2d(
            self.fpn_dim, self.fpn_dim, kernel_size=3, padding=1
        )

        # =====================================================================
        # 3. Classification Head (Study Level)
        # =====================================================================
        # Input: Concatenated GAP vectors from P3, P4, P5
        # Dim: 256 * 3 = 768
        self.cls_head = nn.Linear(self.fpn_dim * 3, Config.NUM_STUDY_CLASSES)

        # =====================================================================
        # 4. Segmentation Head (Image Level)
        # =====================================================================
        # Input: Concatenated features from P2, P3, P4, P5 (all upsampled to P2 size)
        # Dim: 256 * 4 = 1024
        self.seg_conv = nn.Sequential(
            nn.Conv2d(self.fpn_dim * 4, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, Config.NUM_SEG_CLASSES, kernel_size=1),
        )

        # Initialize custom layers
        self._init_weights()

    def _init_weights(self):
        """
        Initialize FPN and Head layers using Kaiming Normal.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input images (B, 3, H, W)

        Returns:
            cls_logits (torch.Tensor): Study predictions (B, 4)
            seg_logits (torch.Tensor): Segmentation mask (B, 1, H, W)
        """
        # --- Encoder (Bottom-Up) ---
        # Stem
        x0 = self.conv1(x)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)
        x0 = self.maxpool(x0)

        # Stages
        c1 = self.layer1(x0)  # Stride 4
        c2 = self.layer2(c1)  # Stride 8
        c3 = self.layer3(c2)  # Stride 16
        c4 = self.layer4(c3)  # Stride 32

        # --- FPN (Top-Down) ---
        # P5
        p5 = self.lat_layer4(c4)

        # P4
        p5_upsampled = F.interpolate(p5, size=c3.shape[-2:], mode="nearest")
        p4 = self.lat_layer3(c3) + p5_upsampled

        # P3
        p4_upsampled = F.interpolate(p4, size=c2.shape[-2:], mode="nearest")
        p3 = self.lat_layer2(c2) + p4_upsampled

        # P2
        p3_upsampled = F.interpolate(p3, size=c1.shape[-2:], mode="nearest")
        p2 = self.lat_layer1(c1) + p3_upsampled

        # Smooth
        p5 = self.smooth_layer4(p5)
        p4 = self.smooth_layer3(p4)
        p3 = self.smooth_layer2(p3)
        p2 = self.smooth_layer1(p2)

        # --- Classification Head ---
        # GAP on P3, P4, P5
        gap_p3 = F.adaptive_avg_pool2d(p3, (1, 1)).flatten(1)
        gap_p4 = F.adaptive_avg_pool2d(p4, (1, 1)).flatten(1)
        gap_p5 = F.adaptive_avg_pool2d(p5, (1, 1)).flatten(1)

        # Concatenate and Linear Projection
        cls_feat = torch.cat([gap_p3, gap_p4, gap_p5], dim=1)
        cls_logits = self.cls_head(cls_feat)

        # --- Segmentation Head ---
        # Target resolution is P2 (Stride 4)
        target_size = p2.shape[-2:]

        # Upsample all levels to P2 resolution
        s5 = F.interpolate(p5, size=target_size, mode="bilinear", align_corners=False)
        s4 = F.interpolate(p4, size=target_size, mode="bilinear", align_corners=False)
        s3 = F.interpolate(p3, size=target_size, mode="bilinear", align_corners=False)
        s2 = p2

        # Concatenate and Convolve
        seg_feat = torch.cat([s2, s3, s4, s5], dim=1)
        seg_logits = self.seg_conv(seg_feat)

        # Final Upsample to Input Resolution (Stride 4 -> Stride 1)
        seg_logits = F.interpolate(
            seg_logits, size=x.shape[-2:], mode="bilinear", align_corners=False
        )

        return cls_logits, seg_logits
