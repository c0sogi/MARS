import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights


class ConvBlock(nn.Module):
    """
    Standard Convolutional Block: Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> ReLU
    """

    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class DepthConditionedUNetPlusPlus(nn.Module):
    """
    U-Net++ with ResNet34 Encoder and Depth Injection.

    Architecture:
    - Encoder: ResNet34 (pretrained) + Custom Input Block (L0)
    - Bottleneck: Depth injection via concatenation
    - Decoder: Nested dense skip pathways (U-Net++)
    - Output: Deep Supervision (multiple heads)
    """

    def __init__(self, num_classes=1, deep_supervision=True):
        super().__init__()
        self.deep_supervision = deep_supervision

        # --- Encoder (ResNet34) ---
        # Load pretrained weights
        self.resnet = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)

        # Modify first layer to accept 1 channel instead of 3
        # We sum the weights of the original 3 channels to preserve initialization magnitude
        old_conv = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        with torch.no_grad():
            self.resnet.conv1.weight.copy_(old_conv.weight.sum(dim=1, keepdim=True))

        # Define Channel counts for Encoder Nodes X(i, 0)
        # L0: 32  (Custom Input Block, 128x128)
        # L1: 64  (ResNet conv1, 64x64)
        # L2: 64  (ResNet layer1, 32x32)
        # L3: 128 (ResNet layer2, 16x16)
        # L4: 256 (ResNet layer3, 8x8)
        # L5: 512 (ResNet layer4, 4x4) - Bottleneck

        self.filters = [32, 64, 64, 128, 256, 512]

        # L0 Input Block (Preserves 128x128 resolution)
        self.input_block = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # --- Depth Injection ---
        # Projects scalar depth to match bottleneck channels
        self.depth_projector = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.Sigmoid(),  # Normalize influence
        )
        # Fusion block after concatenation (512 + 512 -> 512)
        self.bottleneck_fusion = ConvBlock(self.filters[5] + 512, self.filters[5])

        # --- Decoder (Nested Skip Pathways) ---
        # Notation: conv{i}_{j} produces node X(i, j)
        # Input channels = sum(channels of X(i, k) for k < j) + channels of Up(X(i+1, j-1))

        # Column J=1
        self.conv4_1 = ConvBlock(self.filters[4] + self.filters[5], self.filters[4])
        self.conv3_1 = ConvBlock(self.filters[3] + self.filters[4], self.filters[3])
        self.conv2_1 = ConvBlock(self.filters[2] + self.filters[3], self.filters[2])
        self.conv1_1 = ConvBlock(self.filters[1] + self.filters[2], self.filters[1])
        self.conv0_1 = ConvBlock(self.filters[0] + self.filters[1], self.filters[0])

        # Column J=2
        self.conv3_2 = ConvBlock(self.filters[3] * 2 + self.filters[4], self.filters[3])
        self.conv2_2 = ConvBlock(self.filters[2] * 2 + self.filters[3], self.filters[2])
        self.conv1_2 = ConvBlock(self.filters[1] * 2 + self.filters[2], self.filters[1])
        self.conv0_2 = ConvBlock(self.filters[0] * 2 + self.filters[1], self.filters[0])

        # Column J=3
        self.conv2_3 = ConvBlock(self.filters[2] * 3 + self.filters[3], self.filters[2])
        self.conv1_3 = ConvBlock(self.filters[1] * 3 + self.filters[2], self.filters[1])
        self.conv0_3 = ConvBlock(self.filters[0] * 3 + self.filters[1], self.filters[0])

        # Column J=4
        self.conv1_4 = ConvBlock(self.filters[1] * 4 + self.filters[2], self.filters[1])
        self.conv0_4 = ConvBlock(self.filters[0] * 4 + self.filters[1], self.filters[0])

        # --- Deep Supervision Heads ---
        self.final1 = nn.Conv2d(self.filters[0], num_classes, kernel_size=1)
        self.final2 = nn.Conv2d(self.filters[0], num_classes, kernel_size=1)
        self.final3 = nn.Conv2d(self.filters[0], num_classes, kernel_size=1)
        self.final4 = nn.Conv2d(self.filters[0], num_classes, kernel_size=1)

    def _up(self, x, target_tensor):
        """Upsamples x to match target_tensor spatial dimensions."""
        return F.interpolate(
            x, size=target_tensor.shape[2:], mode="bilinear", align_corners=True
        )

    def forward(self, x, depth):
        # --- Encoder Forward ---
        # L0: 128x128
        x0_0 = self.input_block(x)

        # L1: 64x64
        x1_0 = self.resnet.conv1(x)
        x1_0 = self.resnet.bn1(x1_0)
        x1_0 = self.resnet.relu(x1_0)

        # L2: 32x32
        x_pool = self.resnet.maxpool(x1_0)
        x2_0 = self.resnet.layer1(x_pool)

        # L3: 16x16
        x3_0 = self.resnet.layer2(x2_0)

        # L4: 8x8
        x4_0 = self.resnet.layer3(x3_0)

        # L5: 4x4 (Bottleneck)
        x5_0 = self.resnet.layer4(x4_0)

        # --- Depth Injection ---
        # Project depth: (N, 1) -> (N, 512) -> (N, 512, 1, 1)
        d = self.depth_projector(depth).unsqueeze(-1).unsqueeze(-1)
        # Expand to spatial dims of bottleneck: (N, 512, 4, 4)
        d = d.expand_as(x5_0)
        # Concatenate and Fuse
        x5_0 = torch.cat([x5_0, d], dim=1)
        x5_0 = self.bottleneck_fusion(x5_0)

        # --- Decoder Forward ---
        # Column J=1
        x4_1 = self.conv4_1(torch.cat([x4_0, self._up(x5_0, x4_0)], 1))
        x3_1 = self.conv3_1(torch.cat([x3_0, self._up(x4_0, x3_0)], 1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self._up(x3_0, x2_0)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self._up(x2_0, x1_0)], 1))
        x0_1 = self.conv0_1(torch.cat([x0_0, self._up(x1_0, x0_0)], 1))

        # Column J=2
        x3_2 = self.conv3_2(torch.cat([x3_0, x3_1, self._up(x4_1, x3_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self._up(x3_1, x2_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self._up(x2_1, x1_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self._up(x1_1, x0_0)], 1))

        # Column J=3
        x2_3 = self.conv2_3(torch.cat([x2_0, x2_1, x2_2, self._up(x3_2, x2_0)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self._up(x2_2, x1_0)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self._up(x1_2, x0_0)], 1))

        # Column J=4
        x1_4 = self.conv1_4(
            torch.cat([x1_0, x1_1, x1_2, x1_3, self._up(x2_3, x1_0)], 1)
        )
        x0_4 = self.conv0_4(
            torch.cat([x0_0, x0_1, x0_2, x0_3, self._up(x1_3, x0_0)], 1)
        )

        # --- Output Heads ---
        if self.deep_supervision:
            out1 = self.final1(x0_1)
            out2 = self.final2(x0_2)
            out3 = self.final3(x0_3)
            out4 = self.final4(x0_4)
            return [out1, out2, out3, out4]
        else:
            return self.final4(x0_4)
