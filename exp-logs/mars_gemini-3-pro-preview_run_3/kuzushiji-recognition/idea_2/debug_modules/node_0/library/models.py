import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from library.config import Config


class CenterNetDetector(nn.Module):
    """
    Stage 1: Class-Agnostic Keypoint Detector.
    Uses ResNet-18 backbone with FPN to detect character centers, sizes, and offsets.
    Output resolution is stride 4 (1/4 of input size).
    """

    def __init__(self, pretrained=True):
        super(CenterNetDetector, self).__init__()

        # Load Backbone (ResNet18)
        # We use weights parameter for newer torchvision versions
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)

        # FPN Lateral Layers
        # Project all backbone levels to 256 channels
        # ResNet18 channel sizes: layer1=64, layer2=128, layer3=256, layer4=512
        self.lat4 = nn.Conv2d(512, 256, kernel_size=1)
        self.lat3 = nn.Conv2d(256, 256, kernel_size=1)
        self.lat2 = nn.Conv2d(128, 256, kernel_size=1)
        self.lat1 = nn.Conv2d(64, 256, kernel_size=1)

        # Final smoothing layer to reduce aliasing after fusion
        self.smooth = nn.Conv2d(256, 128, kernel_size=3, padding=1)

        # Heads
        # 1. Heatmap Head: Binary classification (text vs background)
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid(),  # Output probability [0, 1]
        )

        # 2. Size Head: Regress Width and Height
        self.wh_head = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=1),  # Output w, h
        )

        # 3. Offset Head: Regress local x, y offsets
        self.offset_head = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=1),  # Output dx, dy
        )

    def forward(self, x):
        """
        Args:
            x: Input images (B, 3, H, W)
        Returns:
            dict containing 'heatmap', 'size_map', 'offset_map'
        """
        # Bottom-up pathway (Backbone)
        # Stem
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)  # Stride 4 so far

        # Layers
        c1 = self.backbone.layer1(x)  # 64, Stride 4
        c2 = self.backbone.layer2(c1)  # 128, Stride 8
        c3 = self.backbone.layer3(c2)  # 256, Stride 16
        c4 = self.backbone.layer4(c3)  # 512, Stride 32

        # Top-down pathway (FPN)
        p4 = self.lat4(c4)

        p3 = self.lat3(c3) + F.interpolate(p4, scale_factor=2, mode="nearest")
        p2 = self.lat2(c2) + F.interpolate(p3, scale_factor=2, mode="nearest")
        p1 = self.lat1(c1) + F.interpolate(p2, scale_factor=2, mode="nearest")

        # Final feature map at stride 4
        out = self.smooth(p1)

        # Heads
        heatmap = self.heatmap_head(out)
        size_map = self.wh_head(out)
        offset_map = self.offset_head(out)

        return {"heatmap": heatmap, "size_map": size_map, "offset_map": offset_map}


class CharacterClassifier(nn.Module):
    """
    Stage 2: Dedicated Character Classifier.
    Uses ResNet-34 to classify cropped character images into one of the Unicode classes.
    """

    def __init__(self, num_classes=Config.NUM_CLASSES, pretrained=True):
        super(CharacterClassifier, self).__init__()

        # Load Backbone (ResNet34)
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet34(weights=weights)

        # Replace the final Fully Connected layer
        # ResNet34 fc input features is 512
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        """
        Args:
            x: Input crops (B, 3, 64, 64)
        Returns:
            logits: (B, num_classes)
        """
        # ResNet forward pass
        # Note: ResNet uses AdaptiveAvgPool2d((1,1)) before the FC layer,
        # so it handles the 64x64 input size automatically (resulting in 1x1 feature map).
        return self.backbone(x)
