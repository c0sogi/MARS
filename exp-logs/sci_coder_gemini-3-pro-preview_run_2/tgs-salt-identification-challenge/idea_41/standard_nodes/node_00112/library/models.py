import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from library.config import Config


class ResNet34Encoder(nn.Module):
    """
    ResNet34 Encoder modified for 1-channel input.
    Returns features from all stages for skip connections.
    """

    def __init__(self, pretrained=True):
        super(ResNet34Encoder, self).__init__()

        # Load weights if requested
        weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        self.resnet = models.resnet34(weights=weights)

        # Modify first convolution for 1-channel input
        # Original: (64, 3, 7, 7)
        original_conv = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(
            1,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None,
        )

        # Sum weights across channel dimension to preserve filters
        with torch.no_grad():
            self.resnet.conv1.weight.data = original_conv.weight.data.sum(
                dim=1, keepdim=True
            )

        # Remove classification head
        del self.resnet.fc
        del self.resnet.avgpool

    def forward(self, x):
        # Stage 0
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x0 = self.resnet.relu(x)  # 64, H/2

        # Stage 1
        x = self.resnet.maxpool(x0)
        x1 = self.resnet.layer1(x)  # 64, H/4

        # Stage 2
        x2 = self.resnet.layer2(x1)  # 128, H/8

        # Stage 3
        x3 = self.resnet.layer3(x2)  # 256, H/16

        # Stage 4 (Bottleneck)
        x4 = self.resnet.layer4(x3)  # 512, H/32

        return [x0, x1, x2, x3, x4]


class DecoderBlock(nn.Module):
    """
    LinkNet Decoder Block: 1x1 Reduce -> 3x3 Deconv -> 1x1 Expand.
    """

    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()

        # Internal width is in_channels // 4 as per LinkNet paper/description
        mid_channels = max(in_channels // 4, 32)

        self.block = nn.Sequential(
            # 1x1 Conv (Reduce)
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # 3x3 Transpose Conv (Upsample)
            nn.ConvTranspose2d(
                mid_channels,
                mid_channels,
                3,
                stride=2,
                padding=1,
                output_padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # 1x1 Conv (Expand)
            nn.Conv2d(mid_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class WideLinkNetDecoder(nn.Module):
    """
    Wide LinkNet Decoder with additive skip connections.
    """

    def __init__(self, encoder_channels, bottleneck_channels=None):
        super(WideLinkNetDecoder, self).__init__()

        # encoder_channels: [64, 64, 128, 256, 512] (x0 to x4)
        if bottleneck_channels is None:
            bottleneck_channels = encoder_channels[-1]

        # Decoder 4: x4 (bottleneck) -> x3 size
        self.dec4 = DecoderBlock(bottleneck_channels, encoder_channels[-2])

        # Decoder 3: x3 size -> x2 size
        self.dec3 = DecoderBlock(encoder_channels[-2], encoder_channels[-3])

        # Decoder 2: x2 size -> x1 size
        self.dec2 = DecoderBlock(encoder_channels[-3], encoder_channels[-4])

        # Decoder 1: x1 size -> x0 size
        self.dec1 = DecoderBlock(encoder_channels[-4], encoder_channels[-5])

        # Final upsampling to restore original H, W
        # x0 is H/2. We need H.
        self.final = nn.Sequential(
            nn.ConvTranspose2d(
                encoder_channels[-5],
                32,
                3,
                stride=2,
                padding=1,
                output_padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),  # Logits
        )

    def forward(self, features):
        x0, x1, x2, x3, x4 = features

        # Additive Skip Connections

        # Block 4
        d4 = self.dec4(x4)
        d4 = d4 + x3

        # Block 3
        d3 = self.dec3(d4)
        d3 = d3 + x2

        # Block 2
        d2 = self.dec2(d3)
        d2 = d2 + x1

        # Block 1
        d1 = self.dec1(d2)
        d1 = d1 + x0

        # Final
        out = self.final(d1)
        return out


class SpecialistTeacher(nn.Module):
    """
    Teacher model with explicit Depth Injection at the bottleneck.
    """

    def __init__(self):
        super(SpecialistTeacher, self).__init__()
        pretrained = Config.ENCODER_PRETRAINED == "imagenet"
        self.encoder = ResNet34Encoder(pretrained=pretrained)

        # Depth Injector MLP
        # Projects scalar depth to 512 channels to match ResNet34 bottleneck
        self.depth_mlp = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 512),
            nn.ReLU(inplace=True),
        )

        # Decoder
        # Input channels = 512 (Image features) + 512 (Depth features) = 1024
        self.decoder = WideLinkNetDecoder(
            encoder_channels=[64, 64, 128, 256, 512], bottleneck_channels=1024
        )

    def forward(self, x, z):
        # x: Image (B, 1, H, W)
        # z: Depth (B, 1)

        features = self.encoder(x)
        x4 = features[-1]  # (B, 512, H/32, W/32)

        # Project depth
        z_feat = self.depth_mlp(z)  # (B, 512)

        # Expand spatially
        z_feat = z_feat.unsqueeze(2).unsqueeze(3)  # (B, 512, 1, 1)
        z_feat = z_feat.expand_as(x4)  # (B, 512, H/32, W/32)

        # Concatenate
        x4_cat = torch.cat([x4, z_feat], dim=1)  # (B, 1024, H/32, W/32)

        # Replace bottleneck in features list for decoder
        features[-1] = x4_cat

        logits = self.decoder(features)
        return logits


class GeneralistStudent(nn.Module):
    """
    Student model without depth injection, but with an auxiliary depth head.
    """

    def __init__(self):
        super(GeneralistStudent, self).__init__()
        pretrained = Config.ENCODER_PRETRAINED == "imagenet"
        self.encoder = ResNet34Encoder(pretrained=pretrained)

        # Auxiliary Depth Head
        # GlobalAvgPool -> MLP -> Scalar
        self.aux_head = nn.Sequential(
            nn.Linear(512, 128), nn.ReLU(inplace=True), nn.Linear(128, 1)
        )

        # Decoder
        # Standard bottleneck channels (512)
        self.decoder = WideLinkNetDecoder(
            encoder_channels=[64, 64, 128, 256, 512], bottleneck_channels=512
        )

    def forward(self, x):
        features = self.encoder(x)
        x4 = features[-1]

        logits = self.decoder(features)

        if self.training:
            # Calculate auxiliary depth prediction
            x4_pool = F.adaptive_avg_pool2d(x4, (1, 1))
            x4_flat = x4_pool.flatten(1)
            depth_pred = self.aux_head(x4_flat)
            return logits, depth_pred
        else:
            return logits
