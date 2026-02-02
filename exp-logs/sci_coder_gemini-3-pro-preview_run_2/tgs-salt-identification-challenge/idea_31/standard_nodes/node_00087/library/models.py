import torch
import torch.nn as nn
import torchvision.models as models
from library.config import Config


class DecoderBlock(nn.Module):
    """
    LinkNet-style Decoder Block with internal width expansion.
    Structure: 1x1 Conv -> Transposed Conv -> 1x1 Conv.
    """

    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()

        # Wide-LinkNet strategy: internal width = in_channels // 4
        # We ensure a minimum width to prevent collapse in edge cases
        mid_channels = max(in_channels // 4, 16)

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
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
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SaltNet(nn.Module):
    """
    ResNet34-based Wide-LinkNet for Salt Segmentation with Depth Injection.
    """

    def __init__(self):
        super(SaltNet, self).__init__()

        # Load ResNet34 backbone
        # Using default weights (ImageNet)
        base = models.resnet34(weights="DEFAULT")

        # Modify first convolution for 1-channel input (Grayscale)
        # We sum the weights of the original RGB filters to preserve structural kernels
        original_conv1 = base.conv1
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.conv1.weight.data = original_conv1.weight.data.sum(dim=1, keepdim=True)

        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool

        # Encoder Layers
        self.layer1 = base.layer1  # 64 channels
        self.layer2 = base.layer2  # 128 channels
        self.layer3 = base.layer3  # 256 channels
        self.layer4 = base.layer4  # 512 channels

        # Depth Injection Module
        self.depth_emb_dim = 64
        self.depth_mlp = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, self.depth_emb_dim),
            nn.ReLU(inplace=True),
        )

        # Decoder Construction
        # Input to dec4 is 512 (encoder) + 64 (depth)
        in_ch_dec4 = 512 + self.depth_emb_dim

        self.dec4 = DecoderBlock(in_ch_dec4, 256)
        self.dec3 = DecoderBlock(256, 128)
        self.dec2 = DecoderBlock(128, 64)
        self.dec1 = DecoderBlock(64, 64)

        # Final upsampling block to restore original resolution (H/2 -> H)
        # Input: 64ch (from dec1 + x0 skip), Output: 1ch (Logits)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
        )

    def forward(self, x, depth):
        """
        Forward pass.
        Args:
            x: Input image (B, 1, H, W)
            depth: Depth scalar (B, 1).
        """
        # ---------------------
        # Encoder
        # ---------------------
        x0 = self.conv1(x)  # 64, H/2, W/2
        x0 = self.bn1(x0)
        x0 = self.relu(x0)

        x_pool = self.maxpool(x0)  # 64, H/4, W/4

        x1 = self.layer1(x_pool)  # 64, H/4, W/4
        x2 = self.layer2(x1)  # 128, H/8, W/8
        x3 = self.layer3(x2)  # 256, H/16, W/16
        x4 = self.layer4(x3)  # 512, H/32, W/32

        # ---------------------
        # Bottleneck / Depth Injection
        # ---------------------
        # Project and inject depth
        d = self.depth_mlp(depth)  # (B, 64)
        d = d.unsqueeze(2).unsqueeze(3)  # (B, 64, 1, 1)
        # Expand to match spatial dimensions of x4
        d = d.expand(-1, -1, x4.size(2), x4.size(3))

        # Concatenate depth features with encoder features
        bottleneck = torch.cat([x4, d], dim=1)  # (B, 576, H/32, W/32)

        # ---------------------
        # Decoder (Additive Skips)
        # ---------------------
        d4 = self.dec4(bottleneck)
        d4 = d4 + x3  # Skip from Layer3

        d3 = self.dec3(d4)
        d3 = d3 + x2  # Skip from Layer2

        d2 = self.dec2(d3)
        d2 = d2 + x1  # Skip from Layer1

        d1 = self.dec1(d2)
        d1 = d1 + x0  # Skip from Conv1 (before pooling)

        # Final Upsample to original resolution
        out = self.final(d1)

        return out
