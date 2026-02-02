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


class DeepSupervisedResNet18UNet(nn.Module):
    """
    ResNet18 Encoder + U-Net Decoder with Deep Supervision.

    Outputs:
        - logit_cls: Study-level classification (N, 4)
        - logit_seg_final: Segmentation mask at full resolution (N, 1, H, W)
        - logit_seg_aux1: Segmentation mask at 1/2 resolution (N, 1, H/2, W/2)
        - logit_seg_aux2: Segmentation mask at 1/4 resolution (N, 1, H/4, W/4)
    """

    def __init__(self, pretrained=Config.PRETRAINED):
        super(DeepSupervisedResNet18UNet, self).__init__()

        # 1. Encoder (ResNet18)
        # ---------------------
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet18(weights=weights)

        # Extract layers
        # conv1 output is 64 channels, stride 2 (H/2)
        self.encoder_conv1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)

        # maxpool stride 2 -> H/4
        self.encoder_maxpool = resnet.maxpool

        # layer1: 64 channels, H/4
        self.encoder_layer1 = resnet.layer1
        # layer2: 128 channels, H/8
        self.encoder_layer2 = resnet.layer2
        # layer3: 256 channels, H/16
        self.encoder_layer3 = resnet.layer3
        # layer4: 512 channels, H/32
        self.encoder_layer4 = resnet.layer4

        # 2. Decoder
        # ----------
        # d4: Up(e4) + e3 -> (512 + 256) -> 256. Res: H/16
        self.decoder4 = DecoderBlock(
            in_channels=512, skip_channels=256, out_channels=256
        )

        # d3: Up(d4) + e2 -> (256 + 128) -> 128. Res: H/8
        self.decoder3 = DecoderBlock(
            in_channels=256, skip_channels=128, out_channels=128
        )

        # d2: Up(d3) + e1 -> (128 + 64) -> 64. Res: H/4
        self.decoder2 = DecoderBlock(in_channels=128, skip_channels=64, out_channels=64)

        # d1: Up(d2) + e0 -> (64 + 64) -> 32. Res: H/2
        self.decoder1 = DecoderBlock(in_channels=64, skip_channels=64, out_channels=32)

        # d0: Up(d1) -> (32 + 0) -> 16. Res: H
        # Final upsample block to reach full resolution
        self.decoder0 = DecoderBlock(in_channels=32, skip_channels=0, out_channels=16)

        # 3. Heads
        # --------

        # Classification Head (Study Level)
        # Attached to the bottleneck (e4)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, Config.NUM_STUDY_CLASSES)

        # Segmentation Heads (Image Level)
        # Final resolution (H, W) - Attached to decoder0 output
        self.seg_head_final = nn.Conv2d(16, 1, kernel_size=1)

        # Aux Head 1 (H/2, W/2) - Attached to decoder1 output
        self.seg_head_aux1 = nn.Conv2d(32, 1, kernel_size=1)

        # Aux Head 2 (H/4, W/4) - Attached to decoder2 output
        self.seg_head_aux2 = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        # -------
        e0 = self.encoder_conv1(x)  # (B, 64, H/2, W/2)
        x_mp = self.encoder_maxpool(e0)
        e1 = self.encoder_layer1(x_mp)  # (B, 64, H/4, W/4)
        e2 = self.encoder_layer2(e1)  # (B, 128, H/8, W/8)
        e3 = self.encoder_layer3(e2)  # (B, 256, H/16, W/16)
        e4 = self.encoder_layer4(e3)  # (B, 512, H/32, W/32)

        # Classification Branch
        # ---------------------
        cls_feat = self.avgpool(e4)
        cls_feat = torch.flatten(cls_feat, 1)
        logit_cls = self.fc(cls_feat)

        # Decoder Branch
        # --------------
        d4 = self.decoder4(e4, e3)  # (B, 256, H/16, W/16)
        d3 = self.decoder3(d4, e2)  # (B, 128, H/8, W/8)

        d2 = self.decoder2(d3, e1)  # (B, 64, H/4, W/4)
        logit_seg_aux2 = self.seg_head_aux2(d2)  # Aux Head 2 (1/4 res)

        d1 = self.decoder1(d2, e0)  # (B, 32, H/2, W/2)
        logit_seg_aux1 = self.seg_head_aux1(d1)  # Aux Head 1 (1/2 res)

        d0 = self.decoder0(d1)  # (B, 16, H, W)
        logit_seg_final = self.seg_head_final(d0)  # Final Head (Full res)

        return logit_cls, logit_seg_final, logit_seg_aux1, logit_seg_aux2
