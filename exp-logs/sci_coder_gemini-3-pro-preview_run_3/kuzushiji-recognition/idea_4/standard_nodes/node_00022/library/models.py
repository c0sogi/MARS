import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from library.config import Config


class CenterNetDetector(nn.Module):
    """
    CenterNet object detector with ResNet-34 backbone and FPN neck.
    Designed for native resolution patches (e.g., 1024x1024 input -> 256x256 output).
    """

    def __init__(self, pretrained=True):
        super(CenterNetDetector, self).__init__()

        # Load Backbone
        weights = "DEFAULT" if pretrained else None
        self.backbone = models.resnet34(weights=weights)

        # FPN Lateral Connections (1x1 Convs to reduce/align channels to 256)
        # ResNet34 channels: layer1=64, layer2=128, layer3=256, layer4=512
        self.lat_layer1 = nn.Conv2d(64, 256, kernel_size=1)
        self.lat_layer2 = nn.Conv2d(128, 256, kernel_size=1)
        self.lat_layer3 = nn.Conv2d(256, 256, kernel_size=1)
        self.lat_layer4 = nn.Conv2d(512, 256, kernel_size=1)

        # FPN Smoothing (3x3 Conv on the final fused features)
        self.smooth_layer = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        # Heads
        # 1. Heatmap Head (Textness/Center probability)
        self.hm_head = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
        )

        # 2. Size Head (Width, Height)
        self.wh_head = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=1),
        )

        # 3. Offset Head (Sub-pixel refinement)
        self.reg_head = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=1),
        )

        self.init_weights()

    def init_weights(self):
        # Initialize head convolution weights
        for head in [self.hm_head, self.wh_head, self.reg_head]:
            for m in head.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, std=0.001)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

        # Initialize heatmap bias to -2.19 (corresponds to sigmoid(x) ~= 0.1)
        # This prevents the initial loss from being dominated by the background
        self.hm_head[-1].bias.data.fill_(-2.19)

    def forward(self, x):
        # --- Backbone Forward ---
        # Input: (B, 3, H, W)
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)  # Stride 4

        c2 = self.backbone.layer1(x)  # Stride 4, 64 ch
        c3 = self.backbone.layer2(c2)  # Stride 8, 128 ch
        c4 = self.backbone.layer3(c3)  # Stride 16, 256 ch
        c5 = self.backbone.layer4(c4)  # Stride 32, 512 ch

        # --- FPN Forward (Top-Down Pathway) ---
        # P5
        p5 = self.lat_layer4(c5)

        # P4
        p5_upsampled = F.interpolate(p5, size=c4.shape[2:], mode="nearest")
        p4 = self.lat_layer3(c4) + p5_upsampled

        # P3
        p4_upsampled = F.interpolate(p4, size=c3.shape[2:], mode="nearest")
        p3 = self.lat_layer2(c3) + p4_upsampled

        # P2 (Target Stride 4)
        p3_upsampled = F.interpolate(p3, size=c2.shape[2:], mode="nearest")
        p2 = self.lat_layer1(c2) + p3_upsampled

        # Smooth the final feature map
        out_feat = self.smooth_layer(p2)

        # --- Heads Forward ---
        hm = self.hm_head(out_feat)
        hm = torch.sigmoid(hm)  # Bound heatmap to [0, 1]

        wh = self.wh_head(out_feat)

        reg = self.reg_head(out_feat)

        return hm, wh, reg


class ResNetClassifier(nn.Module):
    """
    ResNet-50 Image Classifier.
    Used for the second stage to classify crops proposed by the detector.
    """

    def __init__(self, num_classes, pretrained=True):
        super(ResNetClassifier, self).__init__()

        weights = "DEFAULT" if pretrained else None
        self.backbone = models.resnet50(weights=weights)

        # Replace the final fully connected layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)
