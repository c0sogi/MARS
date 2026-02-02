import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet34_Weights, ResNet50_Weights
from library.config import Config


class CenterNetDetector(nn.Module):
    """
    Stage 1: Global Context Detector
    Backbone: ResNet-34 with FPN
    Heads: Heatmap (Textness), Offset, Size
    Input: (B, 3, H, W)
    Output: Heatmap, Size, Offset
    """

    def __init__(self, pretrained=True):
        super(CenterNetDetector, self).__init__()

        # Load Backbone
        weights = ResNet34_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet34(weights=weights)

        # Feature Pyramid Network (FPN) Configuration
        # We project all feature maps to a common channel size
        self.fpn_channels = 64

        # ResNet34 Channel counts:
        # layer1 (c2): 64, layer2 (c3): 128, layer3 (c4): 256, layer4 (c5): 512
        self.lat4 = nn.Conv2d(512, self.fpn_channels, kernel_size=1)
        self.lat3 = nn.Conv2d(256, self.fpn_channels, kernel_size=1)
        self.lat2 = nn.Conv2d(128, self.fpn_channels, kernel_size=1)
        self.lat1 = nn.Conv2d(64, self.fpn_channels, kernel_size=1)

        # Heads
        # 1. Heatmap Head (Textness): 1 channel (Binary: Text vs Background)
        self.head_hm = nn.Sequential(
            nn.Conv2d(self.fpn_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        # 2. Size Head (Width, Height): 2 channels
        self.head_wh = nn.Sequential(
            nn.Conv2d(self.fpn_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=1),
        )

        # 3. Offset Head (Sub-pixel adjustment): 2 channels
        self.head_off = nn.Sequential(
            nn.Conv2d(self.fpn_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=1),
        )

        self._init_head_weights()

    def _init_head_weights(self):
        # Initialize heatmap bias to -2.19 (log(1-pi)/pi where pi=0.1)
        # This prevents the initial loss from exploding due to class imbalance (mostly background)
        for m in self.head_hm.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.constant_(self.head_hm[-2].bias, -2.19)

        # Initialize other heads
        for head in [self.head_wh, self.head_off]:
            for m in head.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, std=0.001)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Bottom-up pathway (ResNet)
        # x: (B, 3, H, W)
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)  # stride 4

        c2 = self.backbone.layer1(x)  # stride 4, 64ch
        c3 = self.backbone.layer2(c2)  # stride 8, 128ch
        c4 = self.backbone.layer3(c3)  # stride 16, 256ch
        c5 = self.backbone.layer4(c4)  # stride 32, 512ch

        # Top-down pathway (FPN)
        p5 = self.lat4(c5)
        p4 = self.lat3(c4) + F.interpolate(p5, scale_factor=2, mode="nearest")
        p3 = self.lat2(c3) + F.interpolate(p4, scale_factor=2, mode="nearest")
        p2 = self.lat1(c2) + F.interpolate(p3, scale_factor=2, mode="nearest")

        # P2 is stride 4, which is the standard output stride for CenterNet

        hm = self.head_hm(p2)
        wh = self.head_wh(p2)
        offset = self.head_off(p2)

        return hm, wh, offset


class ResNetClassifier(nn.Module):
    """
    Stage 2: Verification Classifier
    Backbone: ResNet-50
    Output: 3849 classes (3848 Characters + 1 Background)
    Input: (B, 3, H, W) - usually crops
    Output: (B, Num_Classes)
    """

    def __init__(self, num_classes=Config.NUM_TOTAL_CLASSES, pretrained=True):
        super(ResNetClassifier, self).__init__()

        weights = ResNet50_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet50(weights=weights)

        # Replace the final fully connected layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)
