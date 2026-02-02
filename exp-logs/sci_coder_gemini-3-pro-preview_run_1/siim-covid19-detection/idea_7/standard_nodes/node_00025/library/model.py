import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from library.config import Config


class DecoderBlock(nn.Module):
    """
    Standard U-Net Decoder Block: Upsample -> Concat -> Conv -> BN -> ReLU
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()

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
        # Upsample
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)

        # Concatenate skip connection
        if skip is not None:
            # Ensure dimensions match (handle potential rounding errors in downsampling)
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(
                    x, size=skip.shape[2:], mode="bilinear", align_corners=True
                )
            x = torch.cat([x, skip], dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x


class ResNet18UNet(nn.Module):
    """
    Standard ResNet18 Encoder + U-Net Decoder.
    Simplified to remove Deep Supervision (Cite solution_lesson_node_00008, solution_lesson_node_00021).

    Outputs:
        - logit_cls: Study-level classification (N, 4)
        - logit_seg: Segmentation mask at full resolution (N, 1, H, W)
    """

    def __init__(self, pretrained=Config.PRETRAINED):
        super(ResNet18UNet, self).__init__()

        # 1. Encoder (ResNet18)
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet18(weights=weights)

        self.encoder_conv1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.encoder_maxpool = resnet.maxpool
        self.encoder_layer1 = resnet.layer1
        self.encoder_layer2 = resnet.layer2
        self.encoder_layer3 = resnet.layer3
        self.encoder_layer4 = resnet.layer4

        # 2. Decoder
        self.decoder4 = DecoderBlock(
            in_channels=512, skip_channels=256, out_channels=256
        )
        self.decoder3 = DecoderBlock(
            in_channels=256, skip_channels=128, out_channels=128
        )
        self.decoder2 = DecoderBlock(in_channels=128, skip_channels=64, out_channels=64)
        self.decoder1 = DecoderBlock(in_channels=64, skip_channels=64, out_channels=32)
        self.decoder0 = DecoderBlock(in_channels=32, skip_channels=0, out_channels=16)

        # 3. Heads
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, Config.NUM_STUDY_CLASSES)
        self.seg_head = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        e0 = self.encoder_conv1(x)  # (B, 64, H/2, W/2)
        x_mp = self.encoder_maxpool(e0)
        e1 = self.encoder_layer1(x_mp)  # (B, 64, H/4, W/4)
        e2 = self.encoder_layer2(e1)  # (B, 128, H/8, W/8)
        e3 = self.encoder_layer3(e2)  # (B, 256, H/16, W/16)
        e4 = self.encoder_layer4(e3)  # (B, 512, H/32, W/32)

        # Classification Branch
        cls_feat = self.avgpool(e4)
        cls_feat = torch.flatten(cls_feat, 1)
        logit_cls = self.fc(cls_feat)

        # Decoder Branch
        d4 = self.decoder4(e4, e3)
        d3 = self.decoder3(d4, e2)
        d2 = self.decoder2(d3, e1)
        d1 = self.decoder1(d2, e0)
        d0 = self.decoder0(d1)
        logit_seg = self.seg_head(d0)

        return logit_cls, logit_seg
