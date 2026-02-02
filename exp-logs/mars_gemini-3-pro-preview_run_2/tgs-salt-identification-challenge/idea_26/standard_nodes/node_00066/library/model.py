import torch
import torch.nn as nn
import torchvision.models as models
from library.config import Config


class DecoderBlock(nn.Module):
    """
    Wide-LinkNet Decoder Block.
    Follows the structure: 1x1 Conv -> 3x3 Transposed Conv -> 1x1 Conv.
    The internal dimension is set to in_channels // 4 to maintain width while
    reducing parameter count, characteristic of the 'Wide' variant logic.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Wide-LinkNet logic: internal dimension is derived from input
        internal_channels = in_channels // 4

        self.block = nn.Sequential(
            # 1x1 Conv: Compress/Project
            nn.Conv2d(in_channels, internal_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(internal_channels),
            nn.ReLU(inplace=True),
            # 3x3 Transposed Conv: Upsample
            nn.ConvTranspose2d(
                internal_channels,
                internal_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(internal_channels),
            nn.ReLU(inplace=True),
            # 1x1 Conv: Expand/Project to output
            nn.Conv2d(internal_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResNet34WideLinkNet(nn.Module):
    """
    ResNet34 Wide-LinkNet with Explicit Depth Injection.

    Features:
    1. 1-Channel Input Adaptation.
    2. Explicit Depth Injection at Bottleneck (Concatenation).
       (Cite solution_lesson_node_00041, solution_lesson_node_00024)
    3. Wide-LinkNet Decoder.
    """

    def __init__(self):
        super().__init__()

        # 1. Backbone: ResNet34
        resnet = models.resnet34(pretrained=Config.PRETRAINED)

        # Input Adaptation (1 Channel)
        original_conv1 = resnet.conv1
        self.conv1 = nn.Conv2d(
            Config.IN_CHANNELS,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        with torch.no_grad():
            self.conv1.weight.data = original_conv1.weight.data.sum(dim=1, keepdim=True)

        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        # Encoder Layers
        self.layer1 = resnet.layer1  # 64 ch
        self.layer2 = resnet.layer2  # 128 ch
        self.layer3 = resnet.layer3  # 256 ch
        self.layer4 = resnet.layer4  # 512 ch (Bottleneck)

        # 2. Depth Injection (MLP)
        # Project scalar depth to 32-dim vector (Cite solution_lesson_node_00029)
        self.depth_projector = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
        )

        # 3. Decoder
        # Dec4: Input = Layer4 (512) + Depth (32) = 544. Output = 256.
        self.dec4 = DecoderBlock(512 + 32, 256)

        # Dec3: Input = Dec4 (256). Output = 128.
        self.dec3 = DecoderBlock(256, 128)

        # Dec2: Input = Dec3 (128). Output = 64.
        self.dec2 = DecoderBlock(128, 64)

        # Dec1: Input = Dec2 (64). Output = 64.
        self.dec1 = DecoderBlock(64, 64)

        # Final Upsampling
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x, depth):
        # --- Encoder ---
        x0 = self.conv1(x)
        x0_bn = self.bn1(x0)
        x0_relu = self.relu(x0_bn)
        x1_in = self.maxpool(x0_relu)

        e1 = self.layer1(x1_in)
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        e4 = self.layer4(e3)  # (B, 512, H/32, W/32)

        # --- Depth Injection ---
        # depth shape: (B, 1)
        d_emb = self.depth_projector(depth)  # (B, 32)
        d_emb = d_emb.unsqueeze(2).unsqueeze(3)  # (B, 32, 1, 1)
        # Expand to match feature map spatial dimensions
        d_emb = d_emb.expand(-1, -1, e4.size(2), e4.size(3))

        # Concatenate (Additive Fusion)
        e4_cat = torch.cat([e4, d_emb], dim=1)  # (B, 544, H/32, W/32)

        # --- Decoder ---
        d4 = self.dec4(e4_cat)
        d4 = d4 + e3

        d3 = self.dec3(d4)
        d3 = d3 + e2

        d2 = self.dec2(d3)
        d2 = d2 + e1

        d1 = self.dec1(d2)
        d1 = d1 + x0_relu

        out = self.final_up(d1)
        logits = self.final_conv(out)

        return logits
