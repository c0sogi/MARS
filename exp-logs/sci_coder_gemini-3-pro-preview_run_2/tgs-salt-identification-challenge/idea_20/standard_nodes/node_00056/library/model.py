import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34


class DepthProjector(nn.Module):
    """
    Projects scalar depth z through a Non-Linear MLP to an embedding.
    Structure: Linear -> ReLU -> Linear
    """

    def __init__(self, input_dim=1, hidden_dim=32, output_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        # Ensure input is (B, 1)
        if x.dim() == 1:
            x = x.unsqueeze(1)
        return self.net(x)


class LinkNetDecoderBlock(nn.Module):
    """
    Wide-LinkNet Decoder Block with internal width correction.
    Structure:
      1. 1x1 Conv (in -> in//4)
      2. 3x3 Transposed Conv (in//4 -> in//4, stride=2)
      3. 1x1 Conv (in//4 -> out)
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Width Correction: Calculate internal dimension as in_channels // 4
        internal_channels = in_channels // 4

        self.conv1 = nn.Conv2d(
            in_channels, internal_channels, kernel_size=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(internal_channels)
        self.relu1 = nn.ReLU(inplace=True)

        # Upsampling
        self.deconv2 = nn.ConvTranspose2d(
            internal_channels,
            internal_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(internal_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(
            internal_channels, out_channels, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.deconv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        return x


class ResNet34WideLinkNet(nn.Module):
    """
    ResNet34-WideLinkNet with MLP-Concatenation for Depth Injection.
    """

    def __init__(self):
        super().__init__()

        # 1. Backbone: ResNet34
        # We use pretrained weights for transfer learning
        backbone = resnet34(pretrained=True)

        # 2. Input Adaptation: Modify first conv for 1-channel input
        # Sum weights along the channel dimension: (64, 3, 7, 7) -> (64, 1, 7, 7)
        original_conv1 = backbone.conv1
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.conv1.weight.copy_(original_conv1.weight.sum(dim=1, keepdim=True))

        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        # Encoder Layers
        self.layer1 = (
            backbone.layer1
        )  # Output: 64 channels, 1/4 resolution (32x32 for 128x128 input)
        self.layer2 = backbone.layer2  # Output: 128 channels, 1/8 resolution
        self.layer3 = backbone.layer3  # Output: 256 channels, 1/16 resolution
        self.layer4 = backbone.layer4  # Output: 512 channels, 1/32 resolution

        # 3. Depth Injection Projector
        self.depth_projector = DepthProjector(input_dim=1, hidden_dim=32, output_dim=32)

        # 4. Decoder
        # Bottleneck: Layer4 (512) + Depth Embedding (32) = 544 channels
        self.dec4 = LinkNetDecoderBlock(544, 256)
        self.dec3 = LinkNetDecoderBlock(256, 128)
        self.dec2 = LinkNetDecoderBlock(128, 64)
        self.dec1 = LinkNetDecoderBlock(64, 64)

        # 5. Final Head
        # Upsample from 64x64 (Dec1 output) to 128x128
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),  # Logits
        )

    def forward(self, x, z):
        """
        Args:
            x: Image tensor (B, 1, 128, 128)
            z: Depth tensor (B) or (B, 1)
        """
        # --- Encoder ---
        x0 = self.conv1(x)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)  # (B, 64, 64, 64) -> Skip for Dec1

        x1 = self.maxpool(x0)  # (B, 64, 32, 32)
        x1 = self.layer1(x1)  # (B, 64, 32, 32) -> Skip for Dec2

        x2 = self.layer2(x1)  # (B, 128, 16, 16) -> Skip for Dec3
        x3 = self.layer3(x2)  # (B, 256, 8, 8) -> Skip for Dec4
        x4 = self.layer4(x3)  # (B, 512, 4, 4)

        # --- Depth Injection ---
        d = self.depth_projector(z)  # (B, 32)
        # Expand depth embedding to match spatial dimensions of bottleneck
        d = d.unsqueeze(2).unsqueeze(3)  # (B, 32, 1, 1)
        d = d.expand(-1, -1, x4.size(2), x4.size(3))  # (B, 32, 4, 4)

        # Concatenate
        bottleneck = torch.cat([x4, d], dim=1)  # (B, 544, 4, 4)

        # --- Decoder ---
        # Block 4
        d4 = self.dec4(bottleneck)  # (B, 256, 8, 8)
        d4 = d4 + x3  # Additive Skip Connection

        # Block 3
        d3 = self.dec3(d4)  # (B, 128, 16, 16)
        d3 = d3 + x2

        # Block 2
        d2 = self.dec2(d3)  # (B, 64, 32, 32)
        d2 = d2 + x1

        # Block 1
        d1 = self.dec1(d2)  # (B, 64, 64, 64)
        d1 = d1 + x0

        # --- Final Head ---
        out = self.final_up(d1)  # (B, 1, 128, 128)

        return out
