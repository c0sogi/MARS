import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights
from library.config import IN_CHANNELS


class DecoderBlock(nn.Module):
    """
    Wide-LinkNet Decoder Block.
    Performs dimension reduction, upsampling, and expansion.
    """

    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()

        # LinkNet design: internal width is smaller than input
        internal_channels = in_channels // 4

        # 1x1 Conv to reduce dimensions
        self.conv1 = nn.Conv2d(
            in_channels, internal_channels, kernel_size=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(internal_channels)
        self.relu1 = nn.ReLU(inplace=True)

        # Transposed Conv for upsampling (2x scale)
        self.trans = nn.ConvTranspose2d(
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

        # 1x1 Conv to expand/match target channels
        self.conv2 = nn.Conv2d(
            internal_channels, out_channels, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.trans(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv2(x)
        x = self.bn3(x)
        return x


class SaltNet(nn.Module):
    """
    ResNet34-based Wide-LinkNet for Salt Segmentation.
    Supports two modes:
    1. 'teacher': Injects explicit depth into the bottleneck.
    2. 'student': Regresses depth from the bottleneck (Auxiliary Task).
    """

    def __init__(self, mode="student"):
        super(SaltNet, self).__init__()
        self.mode = mode

        # Load Pretrained Backbone
        weights = ResNet34_Weights.IMAGENET1K_V1
        backbone = resnet34(weights=weights)

        # Modify first conv for 1-channel input
        original_conv1 = backbone.conv1
        self.conv1 = nn.Conv2d(
            IN_CHANNELS, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        with torch.no_grad():
            # Sum weights across RGB channels to initialize grayscale weights
            self.conv1.weight.data = original_conv1.weight.data.sum(dim=1, keepdim=True)

        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        # Encoder Layers
        self.layer1 = backbone.layer1  # 64 ch
        self.layer2 = backbone.layer2  # 128 ch
        self.layer3 = backbone.layer3  # 256 ch
        self.layer4 = backbone.layer4  # 512 ch

        # Channel definitions
        c1, c2, c3, c4 = 64, 128, 256, 512

        # Mode-Specific Modules
        if self.mode == "teacher":
            # Depth Injection Module
            # Projects scalar depth to feature vector and concatenates
            self.depth_channels = 64
            self.depth_injector = nn.Sequential(
                nn.Linear(1, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, self.depth_channels),
                nn.Sigmoid(),
            )
            bottleneck_channels = c4 + self.depth_channels
        else:
            # Auxiliary Depth Head Module
            # Regresses depth from bottleneck features
            self.aux_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(c4, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 1),
            )
            bottleneck_channels = c4

        # Decoder Blocks
        # ResNet34 strides: 4, 8, 16, 32

        # Dec4: Input Bottleneck -> Output 256 (Matches Layer3)
        self.dec4 = DecoderBlock(bottleneck_channels, c3)

        # Dec3: Input 256 -> Output 128 (Matches Layer2)
        self.dec3 = DecoderBlock(c3, c2)

        # Dec2: Input 128 -> Output 64 (Matches Layer1)
        self.dec2 = DecoderBlock(c2, c1)

        # Dec1: Input 64 -> Output 64 (Matches Conv1/Pool output)
        self.dec1 = DecoderBlock(c1, 64)

        # Final Upsampling Block
        # From 64x64 (Dec1 output) to 128x128 (Original Input)
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),  # Logits
        )

    def forward(self, x, depth=None):
        # --- Encoder ---
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x0 = x  # Skip connection 0 (64x64, 64ch)

        x = self.maxpool(x)

        x1 = self.layer1(x)  # Skip connection 1 (32x32, 64ch)
        x2 = self.layer2(x1)  # Skip connection 2 (16x16, 128ch)
        x3 = self.layer3(x2)  # Skip connection 3 (8x8, 256ch)
        x4 = self.layer4(x3)  # Bottleneck (4x4, 512ch)

        # --- Bottleneck Processing ---
        if self.mode == "teacher":
            if depth is None:
                raise ValueError("Teacher mode requires depth input")

            # Inject Depth
            d = self.depth_injector(depth)  # (B, 64)
            d = d.unsqueeze(-1).unsqueeze(-1)  # (B, 64, 1, 1)
            d = d.expand(-1, -1, x4.size(2), x4.size(3))  # Expand to spatial dims
            x4 = torch.cat([x4, d], dim=1)  # (B, 576, 4, 4)

        elif self.mode == "student":
            # Predict Depth (Auxiliary Task)
            pred_depth = self.aux_head(x4)

        # --- Decoder (Additive Skips) ---
        d4 = self.dec4(x4)
        d4 = d4 + x3
        d4 = F.relu(d4)

        d3 = self.dec3(d4)
        d3 = d3 + x2
        d3 = F.relu(d3)

        d2 = self.dec2(d3)
        d2 = d2 + x1
        d2 = F.relu(d2)

        d1 = self.dec1(d2)
        d1 = d1 + x0
        d1 = F.relu(d1)

        # --- Final Head ---
        logits = self.final_up(d1)

        if self.mode == "student":
            return logits, pred_depth
        else:
            return logits
