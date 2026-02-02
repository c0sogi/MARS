import torch
import torch.nn as nn
import torchvision
from torchvision.models import resnet34, ResNet34_Weights


class DecoderBlock(nn.Module):
    """
    Wide-LinkNet Decoder Block.
    Performs upsampling and feature processing.
    Structure: 1x1 Conv (Reduce) -> 3x3 Deconv (Upsample) -> 1x1 Conv (Expand).
    """

    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()

        # Internal width is 1/4 of input channels (Standard LinkNet)
        mid_channels = in_channels // 4

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu = nn.ReLU(inplace=True)

        # Upsampling with stride 2
        self.deconv2 = nn.ConvTranspose2d(
            mid_channels,
            mid_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(mid_channels)

        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.deconv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        return x


class SaltNet(nn.Module):
    """
    ResNet34 + Wide-LinkNet Architecture.
    Supports two modes:
    1. 'teacher': Injects explicit depth information at the bottleneck.
    2. 'student': Predicts depth via an auxiliary head (Multi-Task Learning).
    """

    def __init__(self, mode="student"):
        super(SaltNet, self).__init__()
        self.mode = mode
        assert mode in ["teacher", "student"], "Mode must be 'teacher' or 'student'"

        # Load Pretrained ResNet34
        # Using weights=DEFAULT for best available weights
        weights = ResNet34_Weights.DEFAULT
        self.resnet = resnet34(weights=weights)

        # Modify First Convolution for 1-Channel Input
        # Sum weights across the channel dimension to keep intensity distribution
        original_conv = self.resnet.conv1
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.conv1.weight.copy_(
                torch.sum(original_conv.weight, dim=1, keepdim=True)
            )

        self.bn1 = self.resnet.bn1
        self.relu = self.resnet.relu
        self.maxpool = self.resnet.maxpool

        # Encoder Layers
        self.layer1 = self.resnet.layer1  # 64 channels
        self.layer2 = self.resnet.layer2  # 128 channels
        self.layer3 = self.resnet.layer3  # 256 channels
        self.layer4 = self.resnet.layer4  # 512 channels

        # ---------------------------------------------------------
        # Mode-Specific Components
        # ---------------------------------------------------------
        self.depth_channels = 0

        if self.mode == "teacher":
            # Depth Injector: MLP to project scalar depth to feature channels
            self.depth_channels = 64
            self.depth_mlp = nn.Sequential(
                nn.Linear(1, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, self.depth_channels),
                nn.Sigmoid(),  # Normalize injected features
            )

        if self.mode == "student":
            # Auxiliary Depth Head: Predicts depth from bottleneck
            self.aux_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(512, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 1),
            )

        # ---------------------------------------------------------
        # Decoder (Wide-LinkNet)
        # ---------------------------------------------------------
        # Dec4: Takes Layer4 (512) + Depth Injection (if teacher)
        # Output: 256 (matches Layer3)
        dec4_in = 512 + self.depth_channels
        self.dec4 = DecoderBlock(dec4_in, 256)

        # Dec3: Takes Dec4 (256) -> Output 128 (matches Layer2)
        self.dec3 = DecoderBlock(256, 128)

        # Dec2: Takes Dec3 (128) -> Output 64 (matches Layer1)
        self.dec2 = DecoderBlock(128, 64)

        # Dec1: Takes Dec2 (64) -> Output 64 (matches Conv1 output)
        self.dec1 = DecoderBlock(64, 64)

        # Final Upsample Block
        # Upsamples from 64x64 (Dec1 output) to 128x128 (Original Input)
        self.final = nn.Sequential(
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

    def forward(self, x, depth=None):
        # ---------------------------------------------------------
        # Encoder
        # ---------------------------------------------------------
        x0 = self.conv1(x)  # 128 -> 64 (Stride 2)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)  # Feature map: 64 channels, 64x64

        x1 = self.maxpool(x0)  # 64 -> 32 (Stride 2)

        e1 = self.layer1(x1)  # 64 channels, 32x32
        e2 = self.layer2(e1)  # 128 channels, 16x16
        e3 = self.layer3(e2)  # 256 channels, 8x8
        e4 = self.layer4(e3)  # 512 channels, 4x4

        bottleneck = e4

        # ---------------------------------------------------------
        # Mode-Specific Logic
        # ---------------------------------------------------------
        aux_out = None

        if self.mode == "student":
            # Predict depth from bottleneck
            aux_out = self.aux_head(bottleneck)

        if self.mode == "teacher":
            if depth is None:
                raise ValueError("Depth input is required for Teacher mode.")

            # Inject Depth
            # depth: (N, 1) -> (N, C_depth)
            d_feat = self.depth_mlp(depth)
            # Expand spatially: (N, C_depth, 1, 1) -> (N, C_depth, H, W)
            d_feat = d_feat.unsqueeze(2).unsqueeze(3)
            d_feat = d_feat.expand(-1, -1, bottleneck.size(2), bottleneck.size(3))

            # Concatenate
            bottleneck = torch.cat([bottleneck, d_feat], dim=1)

        # ---------------------------------------------------------
        # Decoder (with Additive Skips)
        # ---------------------------------------------------------
        # Block 4
        d4 = self.dec4(bottleneck)  # 4x4 -> 8x8
        d4 = d4 + e3  # Add Layer3 features

        # Block 3
        d3 = self.dec3(d4)  # 8x8 -> 16x16
        d3 = d3 + e2  # Add Layer2 features

        # Block 2
        d2 = self.dec2(d3)  # 16x16 -> 32x32
        d2 = d2 + e1  # Add Layer1 features

        # Block 1
        d1 = self.dec1(d2)  # 32x32 -> 64x64
        d1 = d1 + x0  # Add Conv1 features (before pooling)

        # Final Upsample
        logits = self.final(d1)  # 64x64 -> 128x128

        if self.mode == "student":
            return logits, aux_out
        else:
            return logits
