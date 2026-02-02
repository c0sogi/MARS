import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights


class DepthProjector(nn.Module):
    """
    Projects scalar depth into a feature vector using a non-linear MLP.
    Used in Teacher mode to inject depth information.
    """

    def __init__(self, input_dim=1, output_dim=64):
        super(DepthProjector, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class AuxDepthHead(nn.Module):
    """
    Auxiliary head to regress depth from the bottleneck features.
    Used in Student mode to force depth-correlated feature extraction.
    """

    def __init__(self, in_channels=512):
        super(AuxDepthHead, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        x = self.pool(x)
        return self.net(x)


class DecoderBlock(nn.Module):
    """
    Wide-LinkNet Decoder Block with Additive Skip Connections.
    Internal width is in_channels // 4.
    """

    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        inter_channels = in_channels // 4

        # 1x1 Conv to compress
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
        )

        # 3x3 Transpose Conv to upsample
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(
                inter_channels,
                inter_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
        )

        # 1x1 Conv to expand
        self.conv2 = nn.Sequential(
            nn.Conv2d(inter_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip=None):
        out = self.conv1(x)
        out = self.deconv(out)
        out = self.conv2(out)

        if skip is not None:
            # Additive skip connection
            out = out + skip
        return out


class SaltNet(nn.Module):
    """
    ResNet34-based LinkNet architecture for Salt Segmentation.
    Supports two modes:
    1. 'teacher': Injects depth information at the bottleneck.
    2. 'student': Regresses depth information from the bottleneck (Auxiliary Task).
    """

    def __init__(self, mode="teacher"):
        super(SaltNet, self).__init__()
        assert mode in ["teacher", "student"], "Mode must be 'teacher' or 'student'"
        self.mode = mode

        # Load Pretrained Backbone
        weights = ResNet34_Weights.IMAGENET1K_V1
        backbone = resnet34(weights=weights)

        # Modify first convolution to accept 1-channel input
        # We sum the weights of the RGB channels to initialize the grayscale filter
        original_conv1 = backbone.conv1
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.conv1.weight.data = original_conv1.weight.data.sum(dim=1, keepdim=True)

        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        # Encoder Layers
        self.layer1 = backbone.layer1  # 64 channels
        self.layer2 = backbone.layer2  # 128 channels
        self.layer3 = backbone.layer3  # 256 channels
        self.layer4 = backbone.layer4  # 512 channels

        # Mode-Specific Components
        if self.mode == "teacher":
            # Project depth to 64 channels and concat to 512 -> 576
            self.depth_projector = DepthProjector(input_dim=1, output_dim=64)
            center_channels = 512 + 64
        else:
            # Student does not take depth input, but has aux head
            self.aux_head = AuxDepthHead(in_channels=512)
            center_channels = 512

        # Decoder Blocks
        # Each block upsamples and adds the corresponding skip connection

        # Bottleneck -> Dec4 (Upsample to Layer3 resolution)
        # Skip: Layer3 (256 ch)
        self.dec4 = DecoderBlock(center_channels, 256)

        # Dec4 -> Dec3 (Upsample to Layer2 resolution)
        # Skip: Layer2 (128 ch)
        self.dec3 = DecoderBlock(256, 128)

        # Dec3 -> Dec2 (Upsample to Layer1 resolution)
        # Skip: Layer1 (64 ch)
        self.dec2 = DecoderBlock(128, 64)

        # Dec2 -> Dec1 (Upsample to Conv1 resolution)
        # Skip: Conv1 output (64 ch)
        self.dec1 = DecoderBlock(64, 64)

        # Final Block: Upsample to original resolution
        # Conv1 has stride 2, so we are currently at H/2, W/2. Need one more upsample.
        self.final_deconv = nn.Sequential(
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),  # Logits
        )

    def forward(self, x, depth=None):
        """
        Args:
            x: Image tensor (B, 1, H, W)
            depth: Depth tensor (B, 1). Required for Teacher, optional/ignored for Student.
        """
        # --- Encoder ---
        x0 = self.conv1(x)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)  # Shape: (B, 64, H/2, W/2)

        x1 = self.maxpool(x0)  # Shape: (B, 64, H/4, W/4)
        x1 = self.layer1(x1)  # Shape: (B, 64, H/4, W/4)

        x2 = self.layer2(x1)  # Shape: (B, 128, H/8, W/8)
        x3 = self.layer3(x2)  # Shape: (B, 256, H/16, W/16)
        x4 = self.layer4(x3)  # Shape: (B, 512, H/32, W/32)

        # --- Center / Bottleneck ---
        if self.mode == "teacher":
            if depth is None:
                raise ValueError("Depth input is required for Teacher mode.")

            # Project depth and expand spatially
            d = self.depth_projector(depth)  # (B, 64)
            d = d.unsqueeze(2).unsqueeze(3)  # (B, 64, 1, 1)
            d = d.expand(-1, -1, x4.size(2), x4.size(3))  # (B, 64, H/32, W/32)

            # Concatenate
            center = torch.cat([x4, d], dim=1)  # (B, 576, H/32, W/32)
            pred_depth = None

        else:  # Student
            center = x4
            # Regress depth
            pred_depth = self.aux_head(center)

        # --- Decoder ---
        d4 = self.dec4(center, skip=x3)  # -> (B, 256, H/16, W/16)
        d3 = self.dec3(d4, skip=x2)  # -> (B, 128, H/8, W/8)
        d2 = self.dec2(d3, skip=x1)  # -> (B, 64, H/4, W/4)
        d1 = self.dec1(d2, skip=x0)  # -> (B, 64, H/2, W/2)

        # --- Final ---
        logits = self.final_deconv(d1)  # -> (B, 1, H, W)

        if self.mode == "student":
            return logits, pred_depth

        return logits
