import torch
import torch.nn as nn
import timm
import math
from library.config import Config


class KuzushijiDetector(nn.Module):
    """
    CenterNet-style detector using HRNet-W32 backbone.
    Maintains high-resolution features (stride 4) for accurate small object detection.
    """

    def __init__(self, pretrained=True):
        super(KuzushijiDetector, self).__init__()

        # 1. Backbone: HRNet-W32
        # We use features_only=True to get feature maps.
        # out_indices=(0,) ensures we only get the stride-4 output (highest resolution).
        self.backbone = timm.create_model(
            Config.DETECTOR_MODEL_NAME,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0,),
        )

        # Get the number of channels for the stride-4 feature map
        # For hrnet_w32, this is typically 32.
        in_channels = self.backbone.feature_info.channels()[0]

        # 2. Heads
        # Common head configuration for CenterNet: 3x3 Conv -> ReLU -> 1x1 Conv
        head_conv = 64  # Intermediate channels

        # Heatmap Head (Class agnostic -> 1 channel)
        self.hm_head = nn.Sequential(
            nn.Conv2d(in_channels, head_conv, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                head_conv,
                Config.DETECTOR_NUM_CLASSES,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.Sigmoid(),
        )

        # Width/Height Head (2 channels)
        self.wh_head = nn.Sequential(
            nn.Conv2d(in_channels, head_conv, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, 2, kernel_size=1, stride=1, padding=0, bias=True),
        )

        # Offset Head (2 channels)
        self.reg_head = nn.Sequential(
            nn.Conv2d(in_channels, head_conv, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, 2, kernel_size=1, stride=1, padding=0, bias=True),
        )

        self._init_weights()

    def _init_weights(self):
        # Initialize head weights
        for head in [self.hm_head, self.wh_head, self.reg_head]:
            for m in head.modules():
                if isinstance(m, nn.Conv2d):
                    # Normal initialization for conv weights
                    nn.init.normal_(m.weight, std=0.001)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

        # Special initialization for Heatmap bias
        # Initialize bias such that the sigmoid output starts around 0.1 (prior probability)
        # b = -log((1 - pi) / pi) where pi = 0.1 => b approx -2.19
        self.hm_head[-2].bias.data.fill_(-2.19)

    def forward(self, x):
        # Backbone returns a list of features, we take the first (stride 4)
        feats = self.backbone(x)
        x = feats[0]

        hm = self.hm_head(x)
        wh = self.wh_head(x)
        reg = self.reg_head(x)

        return hm, wh, reg


class KuzushijiClassifier(nn.Module):
    """
    ResNet-34 classifier for Kuzushiji characters.
    """

    def __init__(self, num_classes, pretrained=True):
        super(KuzushijiClassifier, self).__init__()

        self.model = timm.create_model(
            Config.CLASSIFIER_MODEL_NAME, pretrained=pretrained, num_classes=num_classes
        )

    def forward(self, x):
        return self.model(x)
