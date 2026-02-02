import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from library.config import Config


class DecoderBlock(nn.Module):
    """
    Wide-LinkNet Decoder Block with Additive Skip Connections.
    Structure: Conv1x1 -> BN/ReLU -> TransposedConv3x3 -> BN/ReLU -> Conv1x1 -> BN/ReLU
    """

    def __init__(self, in_channels, out_channels, use_skip=True):
        super(DecoderBlock, self).__init__()
        self.use_skip = use_skip

        # Internal width logic: in_channels // 4 (Wide variant)
        # Ensure a minimum width to avoid bottlenecks in small layers
        mid_channels = max(in_channels // 4, 32)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.trans = nn.Sequential(
            nn.ConvTranspose2d(
                mid_channels,
                mid_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip=None):
        x = self.conv1(x)
        x = self.trans(x)
        x = self.conv2(x)

        if self.use_skip and skip is not None:
            # Additive skip connection
            # Interpolate if there's a slight shape mismatch (e.g. due to padding/cropping)
            if x.size() != skip.size():
                x = F.interpolate(
                    x, size=skip.shape[2:], mode="bilinear", align_corners=False
                )
            x = x + skip

        return x


class ResNet34Encoder(nn.Module):
    """
    ResNet34 Encoder adapted for 1-channel input.
    Returns features [x0, x1, x2, x3, x4] for skip connections.
    """

    def __init__(self, in_channels=1, pretrained=True):
        super(ResNet34Encoder, self).__init__()
        # Load pretrained model
        backbone = models.resnet34(pretrained=pretrained)

        # Modify first layer for 1-channel input if necessary
        if in_channels != 3:
            old_conv = backbone.conv1
            new_conv = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

            with torch.no_grad():
                # Sum weights across channel dimension to preserve intensity info
                # old_conv.weight shape: [64, 3, 7, 7]
                # new_conv.weight shape: [64, 1, 7, 7]
                new_conv.weight.data = old_conv.weight.sum(dim=1, keepdim=True)

            backbone.conv1 = new_conv

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    def forward(self, x):
        # x: (B, 1, H, W)
        x0 = self.relu(self.bn1(self.conv1(x)))  # (B, 64, H/2, W/2)
        x_pool = self.maxpool(x0)  # (B, 64, H/4, W/4)

        x1 = self.layer1(x_pool)  # (B, 64, H/4, W/4)
        x2 = self.layer2(x1)  # (B, 128, H/8, W/8)
        x3 = self.layer3(x2)  # (B, 256, H/16, W/16)
        x4 = self.layer4(x3)  # (B, 512, H/32, W/32)

        return [x0, x1, x2, x3, x4]


class DepthInjector(nn.Module):
    """
    Projects scalar depth to an embedding and concatenates with feature map.
    """

    def __init__(self, out_channels=64):
        super(DepthInjector, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, out_channels),
            nn.ReLU(inplace=True),
        )
        self.out_channels = out_channels

    def forward(self, depth, feature_map):
        # depth: (B, 1)
        # feature_map: (B, C, H, W)

        # Project depth
        d_emb = self.mlp(depth)  # (B, out_channels)

        # Expand spatially to match feature map
        B, C, H, W = feature_map.shape
        d_emb = d_emb.unsqueeze(2).unsqueeze(3).expand(B, self.out_channels, H, W)

        # Concatenate along channel dimension
        return torch.cat([feature_map, d_emb], dim=1)


class SpecialistTeacher(nn.Module):
    """
    Teacher model that explicitly uses depth information.
    Architecture: ResNet34 + Depth Injection + Wide-LinkNet Decoder.
    """

    def __init__(self):
        super(SpecialistTeacher, self).__init__()
        self.encoder = ResNet34Encoder(
            in_channels=Config.IN_CHANNELS, pretrained=Config.PRETRAINED
        )

        # Depth Injector
        self.depth_channels = 64
        self.injector = DepthInjector(out_channels=self.depth_channels)

        # Decoder
        # D4: In(512 + 64) -> Out(256) (Matches Skip x3: 256)
        self.dec4 = DecoderBlock(512 + self.depth_channels, 256)
        # D3: In(256) -> Out(128) (Matches Skip x2: 128)
        self.dec3 = DecoderBlock(256, 128)
        # D2: In(128) -> Out(64) (Matches Skip x1: 64)
        self.dec2 = DecoderBlock(128, 64)
        # D1: In(64) -> Out(64) (Matches Skip x0: 64)
        # Note: x0 is 64ch. We output 64ch to allow addition.
        self.dec1 = DecoderBlock(64, 64)
        # D0: In(64) -> Out(32) (No Skip, final upsample block)
        self.dec0 = DecoderBlock(64, 32, use_skip=False)

        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x, depth):
        # Encoder
        skips = self.encoder(x)
        x0, x1, x2, x3, x4 = skips

        # Inject Depth at Bottleneck (x4)
        x4_aug = self.injector(depth, x4)

        # Decoder with Additive Skips
        d4 = self.dec4(x4_aug, x3)
        d3 = self.dec3(d4, x2)
        d2 = self.dec2(d3, x1)
        d1 = self.dec1(d2, x0)
        d0 = self.dec0(d1)

        logits = self.final_conv(d0)

        return logits


class GeneralistStudent(nn.Module):
    """
    Student model that predicts masks from images only.
    Includes an Auxiliary Depth Head for regularization during training.
    """

    def __init__(self):
        super(GeneralistStudent, self).__init__()
        self.encoder = ResNet34Encoder(
            in_channels=Config.IN_CHANNELS, pretrained=Config.PRETRAINED
        )

        # Aux Depth Head (attached to bottleneck)
        self.aux_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

        # Decoder (Same structure as Teacher, but without depth injection channels)
        # D4: In(512) -> Out(256)
        self.dec4 = DecoderBlock(512, 256)
        # D3: In(256) -> Out(128)
        self.dec3 = DecoderBlock(256, 128)
        # D2: In(128) -> Out(64)
        self.dec2 = DecoderBlock(128, 64)
        # D1: In(64) -> Out(64)
        self.dec1 = DecoderBlock(64, 64)
        # D0: In(64) -> Out(32)
        self.dec0 = DecoderBlock(64, 32, use_skip=False)

        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        skips = self.encoder(x)
        x0, x1, x2, x3, x4 = skips

        # Aux Head Prediction
        depth_pred = self.aux_head(x4)

        # Decoder
        d4 = self.dec4(x4, x3)
        d3 = self.dec3(d4, x2)
        d2 = self.dec2(d3, x1)
        d1 = self.dec1(d2, x0)
        d0 = self.dec0(d1)

        logits = self.final_conv(d0)

        return logits, depth_pred
