import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from library.config import Config


class ConvBlock(nn.Module):
    """
    Standard Convolutional Block for the decoder:
    Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> ReLU
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class ResNet34UNetPlusPlus(nn.Module):
    """
    U-Net++ (Nested U-Net) architecture with a ResNet-34 encoder.

    Encoder Levels:
    - X0,0: Stride 2 (64 ch)
    - X1,0: Stride 4 (64 ch)
    - X2,0: Stride 8 (128 ch)
    - X3,0: Stride 16 (256 ch)
    - X4,0: Stride 32 (512 ch)

    Decoder uses dense skip connections where node X(i,j) receives inputs from:
    - Upsampled X(i+1, j-1)
    - All previous nodes in the same row: X(i, 0), ..., X(i, j-1)
    """

    def __init__(self, in_channels=3, classes=1):
        super().__init__()

        # Load pre-trained ResNet34
        # Note: 'weights' is the modern equivalent of 'pretrained=True'
        encoder = models.resnet34(weights="IMAGENET1K_V1")

        self.encoder_layers = list(encoder.children())

        # --- Encoder Path ---
        # Stage 0: Conv1 -> BN -> ReLU. Output: (B, 64, H/2, W/2)
        self.layer0 = nn.Sequential(
            self.encoder_layers[0], self.encoder_layers[1], self.encoder_layers[2]
        )

        # Stage 1: MaxPool -> Layer1. Output: (B, 64, H/4, W/4)
        self.layer1 = nn.Sequential(self.encoder_layers[3], self.encoder_layers[4])

        # Stage 2: Layer2. Output: (B, 128, H/8, W/8)
        self.layer2 = self.encoder_layers[5]

        # Stage 3: Layer3. Output: (B, 256, H/16, W/16)
        self.layer3 = self.encoder_layers[6]

        # Stage 4: Layer4. Output: (B, 512, H/32, W/32)
        self.layer4 = self.encoder_layers[7]

        # --- Decoder Configuration ---
        # Define channel counts for each row (level) of the U-Net++
        # Matching ResNet channel widths roughly, but keeping decoder consistent
        filters = [64, 64, 128, 256, 512]

        # --- Decoder Nodes (Nested Skip Pathways) ---

        # Column j=1 (First nested layer)
        # Input: Encoder[i] + Up(Encoder[i+1])
        self.conv0_1 = ConvBlock(filters[0] + filters[1], filters[0])
        self.conv1_1 = ConvBlock(filters[1] + filters[2], filters[1])
        self.conv2_1 = ConvBlock(filters[2] + filters[3], filters[2])
        self.conv3_1 = ConvBlock(filters[3] + filters[4], filters[3])

        # Column j=2 (Second nested layer)
        # Input: Encoder[i] + Node[i,1] + Up(Node[i+1,1])
        self.conv0_2 = ConvBlock(filters[0] * 2 + filters[1], filters[0])
        self.conv1_2 = ConvBlock(filters[1] * 2 + filters[2], filters[1])
        self.conv2_2 = ConvBlock(filters[2] * 2 + filters[3], filters[2])

        # Column j=3 (Third nested layer)
        # Input: Encoder[i] + Node[i,1] + Node[i,2] + Up(Node[i+1,2])
        self.conv0_3 = ConvBlock(filters[0] * 3 + filters[1], filters[0])
        self.conv1_3 = ConvBlock(filters[1] * 3 + filters[2], filters[1])

        # Column j=4 (Final nested layer)
        # Input: Encoder[i] + Node[i,1] + Node[i,2] + Node[i,3] + Up(Node[i+1,3])
        self.conv0_4 = ConvBlock(filters[0] * 4 + filters[1], filters[0])

        # Final 1x1 Convolution to project to class space
        self.final_conv = nn.Conv2d(filters[0], classes, kernel_size=1)

    def _up(self, x):
        """Helper for bilinear upsampling"""
        return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)

    def forward(self, x):
        # --- Encoder Forward ---
        x0_0 = self.layer0(x)  # H/2
        x1_0 = self.layer1(x0_0)  # H/4
        x2_0 = self.layer2(x1_0)  # H/8
        x3_0 = self.layer3(x2_0)  # H/16
        x4_0 = self.layer4(x3_0)  # H/32

        # --- Decoder Forward ---

        # Column j=1
        x0_1 = self.conv0_1(torch.cat([x0_0, self._up(x1_0)], dim=1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self._up(x2_0)], dim=1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self._up(x3_0)], dim=1))
        x3_1 = self.conv3_1(torch.cat([x3_0, self._up(x4_0)], dim=1))

        # Column j=2
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self._up(x1_1)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self._up(x2_1)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self._up(x3_1)], dim=1))

        # Column j=3
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self._up(x1_2)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self._up(x2_2)], dim=1))

        # Column j=4
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self._up(x1_3)], dim=1))

        # --- Final Output ---
        logits = self.final_conv(x0_4)

        # Upsample from H/2 to H (Original Input Size)
        logits = self._up(logits)

        return logits
