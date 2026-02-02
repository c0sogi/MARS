import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from library.config import Config


class ConvBlock(nn.Module):
    """
    Standard Convolutional Block: Conv2d -> BatchNorm2d -> ReLU.
    Used in the segmentation head.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=padding, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class FPN(nn.Module):
    """
    Feature Pyramid Network (FPN) with ResNet-18 backbone for 2.5D MRI Segmentation.

    This model takes a 3-channel input (slice-1, slice, slice+1), extracts features
    using a ResNet-18 encoder, builds a feature pyramid, aggregates features from
    all levels, and outputs a segmentation map for 3 classes.
    """

    def __init__(
        self,
        backbone_name=Config.BACKBONE,
        pretrained=Config.PRETRAINED,
        num_classes=Config.NUM_CLASSES,
    ):
        super(FPN, self).__init__()

        # --- 1. Backbone (Encoder) ---
        if backbone_name == "resnet18":
            # Load pretrained ResNet18
            # Note: Input channels=3 aligns with standard ResNet weights
            try:
                # Newer torchvision versions
                weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
                self.backbone = models.resnet18(weights=weights)
            except AttributeError:
                # Older torchvision versions
                self.backbone = models.resnet18(pretrained=pretrained)

            # Channel dimensions for ResNet18 stages:
            # Layer1 (C2): 64, Layer2 (C3): 128, Layer3 (C4): 256, Layer4 (C5): 512
            self.in_channels = [64, 128, 256, 512]
        else:
            raise NotImplementedError(
                f"Backbone '{backbone_name}' is not implemented in this FPN."
            )

        # --- 2. FPN Layers ---
        self.fpn_out_channels = 256

        # Lateral connections (1x1 convs) to reduce feature channels
        self.lat_layer1 = nn.Conv2d(self.in_channels[0], self.fpn_out_channels, 1)
        self.lat_layer2 = nn.Conv2d(self.in_channels[1], self.fpn_out_channels, 1)
        self.lat_layer3 = nn.Conv2d(self.in_channels[2], self.fpn_out_channels, 1)
        self.lat_layer4 = nn.Conv2d(self.in_channels[3], self.fpn_out_channels, 1)

        # Smoothing layers (3x3 convs) to reduce aliasing
        self.smooth_layer1 = nn.Conv2d(
            self.fpn_out_channels, self.fpn_out_channels, 3, padding=1
        )
        self.smooth_layer2 = nn.Conv2d(
            self.fpn_out_channels, self.fpn_out_channels, 3, padding=1
        )
        self.smooth_layer3 = nn.Conv2d(
            self.fpn_out_channels, self.fpn_out_channels, 3, padding=1
        )
        self.smooth_layer4 = nn.Conv2d(
            self.fpn_out_channels, self.fpn_out_channels, 3, padding=1
        )

        # --- 3. Segmentation Head ---
        # We use a Semantic FPN approach: merge all levels at P2 resolution (1/4 scale)
        self.semantic_head = nn.Sequential(
            ConvBlock(self.fpn_out_channels, 128, kernel_size=3, padding=1),
            nn.Dropout2d(0.1),
            nn.Conv2d(128, num_classes, kernel_size=1),
        )

    def _upsample_add(self, x, y):
        """
        Upsamples tensor x to match the spatial dimensions of tensor y,
        then adds them element-wise.
        """
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False) + y

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, 3, Height, Width).

        Returns:
            torch.Tensor: Output probabilities (Batch, Num_Classes, Height, Width).
        """
        # --- Bottom-up Pathway (Backbone) ---
        # Stem
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        c1 = self.backbone.maxpool(x)  # Scale 1/4

        # ResNet Stages
        c2 = self.backbone.layer1(c1)  # Scale 1/4, 64 ch
        c3 = self.backbone.layer2(c2)  # Scale 1/8, 128 ch
        c4 = self.backbone.layer3(c3)  # Scale 1/16, 256 ch
        c5 = self.backbone.layer4(c4)  # Scale 1/32, 512 ch

        # --- Top-down Pathway (FPN) ---
        # P5
        p5 = self.lat_layer4(c5)

        # P4
        p4 = self._upsample_add(p5, self.lat_layer3(c4))

        # P3
        p3 = self._upsample_add(p4, self.lat_layer2(c3))

        # P2
        p2 = self._upsample_add(p3, self.lat_layer1(c2))

        # Smoothing
        p5 = self.smooth_layer4(p5)
        p4 = self.smooth_layer3(p4)
        p3 = self.smooth_layer2(p3)
        p2 = self.smooth_layer1(p2)

        # --- Feature Aggregation ---
        # Upsample all pyramid levels to P2 resolution (1/4 input size)
        _, _, H, W = p2.size()

        p5_up = F.interpolate(p5, size=(H, W), mode="bilinear", align_corners=False)
        p4_up = F.interpolate(p4, size=(H, W), mode="bilinear", align_corners=False)
        p3_up = F.interpolate(p3, size=(H, W), mode="bilinear", align_corners=False)

        # Sum features (Semantic FPN)
        feats = p2 + p3_up + p4_up + p5_up

        # --- Prediction ---
        logits = self.semantic_head(feats)

        # Upsample to original input resolution (4x upsampling)
        logits = F.interpolate(
            logits, scale_factor=4, mode="bilinear", align_corners=False
        )

        return torch.sigmoid(logits)
