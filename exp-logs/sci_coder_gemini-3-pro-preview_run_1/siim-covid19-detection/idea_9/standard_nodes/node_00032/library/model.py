import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from library.config import Config


class SELayer(nn.Module):
    """
    Squeeze-and-Excitation Block.
    Recalibrates channel-wise feature responses by explicitly modelling interdependencies between channels.
    """

    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SEBasicBlockWrapper(nn.Module):
    """
    Wraps a standard ResNet BasicBlock to inject an SE Layer after the block's output.
    This preserves the pre-trained weights of the BasicBlock.
    """

    def __init__(self, block, channels, reduction=16):
        super(SEBasicBlockWrapper, self).__init__()
        self.block = block
        self.se = SELayer(channels, reduction)

    def forward(self, x):
        out = self.block(x)
        out = self.se(out)
        return out


class ResNet18SEEncoder(nn.Module):
    """
    ResNet18 Encoder with Squeeze-and-Excitation blocks injected after each residual block.
    Returns features at multiple scales for U-Net skip connections.
    """

    def __init__(self, pretrained=True):
        super(ResNet18SEEncoder, self).__init__()

        # Load pre-trained ResNet18
        if pretrained:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            resnet = models.resnet18(weights=weights)
        else:
            resnet = models.resnet18(weights=None)

        # Initial layers
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        # Inject SE blocks into layers
        self.layer1 = self._inject_se(resnet.layer1, 64)
        self.layer2 = self._inject_se(resnet.layer2, 128)
        self.layer3 = self._inject_se(resnet.layer3, 256)
        self.layer4 = self._inject_se(resnet.layer4, 512)

    def _inject_se(self, layer, channels):
        """
        Replaces BasicBlocks in a Sequential layer with SEBasicBlockWrappers.
        """
        layers = []
        for block in layer.children():
            layers.append(SEBasicBlockWrapper(block, channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x0 = self.relu(x)  # c1: (B, 64, H/2, W/2)
        x = self.maxpool(x0)

        # Encoder Blocks
        x1 = self.layer1(x)  # c2: (B, 64, H/4, W/4)
        x2 = self.layer2(x1)  # c3: (B, 128, H/8, W/8)
        x3 = self.layer3(x2)  # c4: (B, 256, H/16, W/16)
        x4 = self.layer4(x3)  # c5: (B, 512, H/32, W/32)

        return [x0, x1, x2, x3, x4]


class DecoderBlock(nn.Module):
    """
    Standard U-Net Decoder Block: Upsample -> Concat -> ConvBlock.
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels + skip_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip=None):
        # Bilinear upsampling
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)

        if skip is not None:
            # Handle potential padding issues if dimensions don't match exactly
            if x.size() != skip.size():
                x = F.interpolate(
                    x, size=skip.shape[2:], mode="bilinear", align_corners=True
                )
            x = torch.cat([x, skip], dim=1)

        return self.conv(x)


class UNetDecoder(nn.Module):
    """
    Decoder network recovering spatial resolution from encoder features.
    """

    def __init__(self):
        super(UNetDecoder, self).__init__()

        # Encoder output channels:
        # x4 (Layer4): 512
        # x3 (Layer3): 256
        # x2 (Layer2): 128
        # x1 (Layer1): 64
        # x0 (Conv1):  64

        self.center = nn.Identity()

        # Decoder stages
        # 1. Input: 512 (x4), Skip: 256 (x3) -> Out: 256
        self.dec4 = DecoderBlock(512, 256, 256)

        # 2. Input: 256, Skip: 128 (x2) -> Out: 128
        self.dec3 = DecoderBlock(256, 128, 128)

        # 3. Input: 128, Skip: 64 (x1) -> Out: 64
        self.dec2 = DecoderBlock(128, 64, 64)

        # 4. Input: 64, Skip: 64 (x0) -> Out: 32
        self.dec1 = DecoderBlock(64, 64, 32)

        # 5. Final Upsample to original resolution (no skip available from stem input usually, or just upsample)
        # Input: 32 -> Upsample -> Out: 16
        self.final_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

    def forward(self, features):
        x0, x1, x2, x3, x4 = features

        x = self.center(x4)
        x = self.dec4(x, x3)
        x = self.dec3(x, x2)
        x = self.dec2(x, x1)
        x = self.dec1(x, x0)
        x = self.final_conv(x)

        return x


class MultiTaskModel(nn.Module):
    """
    SE-ResNet18 Multi-Task U-Net.
    Backbone: ResNet18 + SE Blocks.
    Heads:
        - Classification: GAP + Linear (Shallow Head).
        - Segmentation: U-Net Decoder + 1x1 Conv.
    """

    def __init__(self, num_classes=Config.NUM_CLASSES, pretrained=True):
        super(MultiTaskModel, self).__init__()

        self.encoder = ResNet18SEEncoder(pretrained=pretrained)
        self.decoder = UNetDecoder()

        # Classification Head (Shallow)
        # ResNet18 layer4 output is 512 channels
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.cls_head = nn.Linear(512, num_classes)

        # Segmentation Head
        # Decoder output is 16 channels
        self.seg_head = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        features = self.encoder(x)
        # features: [x0(64), x1(64), x2(128), x3(256), x4(512)]
        encoder_final = features[-1]

        # Classification Branch
        cls_feat = self.avg_pool(encoder_final)
        cls_feat = torch.flatten(cls_feat, 1)
        cls_logits = self.cls_head(cls_feat)

        # Segmentation Branch
        seg_feat = self.decoder(features)
        seg_logits = self.seg_head(seg_feat)

        return cls_logits, seg_logits
