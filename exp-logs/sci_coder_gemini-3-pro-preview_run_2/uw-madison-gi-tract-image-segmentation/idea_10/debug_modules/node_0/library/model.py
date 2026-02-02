import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from library.config import Config


class PyramidPoolingModule(nn.Module):
    """
    Pyramid Pooling Module (PPM) as described in PSPNet.
    Aggregates global context information through region-based average pooling
    at different scales.
    """

    def __init__(self, in_channels, sizes=(1, 2, 3, 6)):
        super(PyramidPoolingModule, self).__init__()
        self.stages = nn.ModuleList([])
        self.stages = nn.ModuleList()

        # The output channels for each pooling branch.
        # Typically in_channels / len(sizes).
        out_channels = in_channels // len(sizes)

        for size in sizes:
            self.stages.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(output_size=(size, size)),
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )

    def forward(self, x):
        # x shape: (Batch, C, H, W)
        h, w = x.shape[2], x.shape[3]

        # Apply each pooling branch and upsample to match input spatial resolution
        ppm_outs = [x]
        for stage in self.stages:
            pooled = stage(x)
            upsampled = F.interpolate(
                pooled, size=(h, w), mode="bilinear", align_corners=True
            )
            ppm_outs.append(upsampled)

        # Concatenate original features with pooled features
        return torch.cat(ppm_outs, dim=1)


class ShuffleNetPSPNet(nn.Module):
    """
    2.5D Segmentation model using a ShuffleNetV2 backbone and a Pyramid Pooling Module.
    Designed for lightweight and efficient volumetric context aggregation.
    """

    def __init__(self, num_classes=None, in_channels=3, pretrained=True):
        super(ShuffleNetPSPNet, self).__init__()

        if num_classes is None:
            num_classes = Config.NUM_CLASSES

        # 1. Backbone: ShuffleNetV2
        # We use x1.0 variant. Output of conv5 is 1024 channels.
        weights = "DEFAULT" if pretrained else None
        self.backbone = models.shufflenet_v2_x1_0(weights=weights)

        # Remove the classification head (fc) as we only need feature extraction
        del self.backbone.fc

        # Check input channels. ShuffleNet expects 3.
        # If in_channels != 3, we would need to replace conv1, but Config says 3.
        if in_channels != 3:
            raise ValueError(
                f"ShuffleNetV2 expects 3 input channels, got {in_channels}"
            )

        # Backbone output channels (conv5 output for shufflenet_v2_x1_0)
        self.backbone_out_channels = 1024

        # 2. Context Aggregation: Pyramid Pooling Module
        # PPM scales: 1x1, 2x2, 3x3, 6x6
        self.ppm = PyramidPoolingModule(self.backbone_out_channels, sizes=(1, 2, 3, 6))

        # Calculate total channels after PPM concatenation
        # Original (1024) + 4 branches * (1024 // 4 = 256) = 1024 + 1024 = 2048
        ppm_out_channels = (
            self.backbone_out_channels + (self.backbone_out_channels // 4) * 4
        )

        # 3. Decoder / Classifier
        # Fuse features and project to class logits
        self.decoder = nn.Sequential(
            nn.Conv2d(ppm_out_channels, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1),
        )

    def forward(self, x):
        input_size = x.shape[-2:]  # (H, W)

        # --- Backbone Forward Pass ---
        # We manually call stages to extract features up to conv5
        x = self.backbone.conv1(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.stage2(x)
        x = self.backbone.stage3(x)
        x = self.backbone.stage4(x)
        features = self.backbone.conv5(x)

        # --- Context Aggregation ---
        # Apply PPM
        context_features = self.ppm(features)

        # --- Decoding ---
        logits = self.decoder(context_features)

        # --- Upsampling ---
        # Upsample logits to original input size
        logits = F.interpolate(
            logits, size=input_size, mode="bilinear", align_corners=True
        )

        return logits
