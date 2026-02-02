import torch
import torch.nn as nn
import timm
import math
from library.config import Config


class CenterNet(nn.Module):
    """
    CenterNet architecture for object detection.

    Backbone: ResNet-18 (default)
    Neck: 3x Transposed Convolutions (Upsampling)
    Heads:
        - hm: Heatmap (Class probabilities)
        - wh: Width and Height
        - reg: Local Offset (Regression)
    """

    def __init__(self, backbone_name=None, pretrained=True):
        super(CenterNet, self).__init__()

        self.backbone_name = backbone_name if backbone_name else Config.BACKBONE
        self.num_classes = Config.NUM_CLASSES

        # =====================================================================
        # Backbone
        # =====================================================================
        # Load backbone using timm
        # features_only=True returns a list of feature maps from different stages
        # out_indices=(4,) requests only the final feature map (usually stride 32)
        self.backbone = timm.create_model(
            self.backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(4,),
        )

        # Determine input channels for the neck dynamically
        dummy_input = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            features = self.backbone(dummy_input)
            in_channels = features[0].shape[1]

        # =====================================================================
        # Neck (Upsampling)
        # =====================================================================
        # We need to upsample from Stride 32 to Stride 4 (factor of 8).
        # We use 3 layers of ConvTranspose2d, each with stride 2.
        # Channel reduction: 512 -> 256 -> 128 -> 64

        self.neck = nn.Sequential(
            # Layer 1: Stride 32 -> 16
            nn.ConvTranspose2d(
                in_channels, 256, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # Layer 2: Stride 16 -> 8
            nn.ConvTranspose2d(
                256, 128, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # Layer 3: Stride 8 -> 4
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # =====================================================================
        # Heads
        # =====================================================================
        # All heads share the same input features from the neck (64 channels)

        # Heatmap Head: Predicts class confidence (C channels)
        self.hm_head = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.num_classes, kernel_size=1, bias=True),
        )

        # Size Head: Predicts width and height (2 channels)
        self.wh_head = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=1, bias=True),
        )

        # Offset Head: Predicts local offset (2 channels)
        self.reg_head = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=1, bias=True),
        )

        self._init_weights()

    def _init_weights(self):
        # Initialize Neck
        for m in self.neck.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Initialize Heads
        for head in [self.hm_head, self.wh_head, self.reg_head]:
            for m in head.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, std=0.001)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

        # Special initialization for Heatmap Head bias
        # This prevents a large loss at the beginning of training when using Focal Loss
        # bias = -log((1 - p) / p) where p is a low prior probability (e.g., 0.1)
        self.hm_head[-1].bias.data.fill_(-2.19)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input images of shape (B, 3, H, W)

        Returns:
            dict: Contains 'hm', 'wh', 'reg' tensors.
        """
        # Backbone
        features = self.backbone(x)
        x = features[0]  # Extract the single feature map returned

        # Neck
        x = self.neck(x)

        # Heads
        hm = self.hm_head(x)
        wh = self.wh_head(x)
        reg = self.reg_head(x)

        # Return dictionary
        # Note: 'hm' contains raw logits. Sigmoid should be applied in loss or inference.
        return {"hm": hm, "wh": wh, "reg": reg}
