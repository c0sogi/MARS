import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from library.config import Config


class DecoderBlock(nn.Module):
    """
    Wide-LinkNet Decoder Block.
    Performs upsampling and adds the skip connection from the encoder.
    """

    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()

        # Wide-LinkNet: internal width calculation
        # Standard LinkNet uses in_channels // 4.
        mid_channels = max(in_channels // 4, 16)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.deconv = nn.Sequential(
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

    def forward(self, x, skip):
        """
        Args:
            x: Input feature map from previous decoder block.
            skip: Skip connection feature map from encoder.
        """
        x = self.conv1(x)
        x = self.deconv(x)
        x = self.conv2(x)

        if skip is not None:
            # Handle potential slight shape mismatch due to padding/pooling
            if x.size() != skip.size():
                x = F.interpolate(
                    x, size=skip.shape[2:], mode="bilinear", align_corners=False
                )
            x = x + skip

        return x


class ResNet34WideLinkNet(nn.Module):
    """
    Multi-Task Wide-LinkNet with ResNet34 Backbone.
    Outputs both segmentation mask logits and scalar depth prediction.
    """

    def __init__(self, pretrained=True):
        super(ResNet34WideLinkNet, self).__init__()

        # Load ResNet34 Backbone
        # Handle torchvision version differences for weights
        try:
            from torchvision.models import ResNet34_Weights

            weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            self.resnet = models.resnet34(weights=weights)
        except ImportError:
            self.resnet = models.resnet34(pretrained=pretrained)

        # 1. Input Adaptation: Modify first layer for 1-channel input
        original_conv1 = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias,
        )

        # Initialize 1-channel weights by summing RGB weights
        # This preserves texture detection capabilities of pretrained weights
        with torch.no_grad():
            self.resnet.conv1.weight.data = original_conv1.weight.data.sum(
                dim=1, keepdim=True
            )

        # 2. Encoder Layers (Splitting ResNet)
        self.encoder0 = nn.Sequential(
            self.resnet.conv1, self.resnet.bn1, self.resnet.relu
        )
        self.encoder1 = nn.Sequential(self.resnet.maxpool, self.resnet.layer1)
        self.encoder2 = self.resnet.layer2
        self.encoder3 = self.resnet.layer3
        self.encoder4 = self.resnet.layer4

        # 3. Depth Injection (Cite solution_lesson_node_00024, solution_lesson_node_00029)
        # Project scalar depth (1) to embedding (32) using MLP
        self.depth_projector = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
        )

        # 4. Decoder Layers (LinkNet Style)
        # ResNet34 Channels: L4=512, L3=256, L2=128, L1=64, L0=64
        # Decoder4 input: 512 (Encoder) + 32 (Depth) = 544
        self.decoder4 = DecoderBlock(544, 256)  # Takes L4+Depth, adds L3
        self.decoder3 = DecoderBlock(256, 128)  # Takes D4, adds L2
        self.decoder2 = DecoderBlock(128, 64)  # Takes D3, adds L1
        self.decoder1 = DecoderBlock(64, 64)  # Takes D2, adds L0

        # 5. Final Segmentation Head
        # Upsample from H/2 to H and map to 1 class
        self.final_conv = nn.Sequential(
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
        )

    def forward(self, x, depth):
        # --- Encoder ---
        # Input: (N, 1, H, W)
        e0 = self.encoder0(x)  # (N, 64, H/2, W/2)
        e1 = self.encoder1(e0)  # (N, 64, H/4, W/4)
        e2 = self.encoder2(e1)  # (N, 128, H/8, W/8)
        e3 = self.encoder3(e2)  # (N, 256, H/16, W/16)
        e4 = self.encoder4(e3)  # (N, 512, H/32, W/32)

        # --- Depth Injection ---
        # depth: (N, 1)
        d_emb = self.depth_projector(depth)  # (N, 32)
        d_emb = d_emb.unsqueeze(2).unsqueeze(3)  # (N, 32, 1, 1)
        # Expand to match feature map spatial dimensions
        d_emb = d_emb.expand(-1, -1, e4.size(2), e4.size(3))  # (N, 32, H/32, W/32)

        # Concatenate (Cite solution_lesson_node_00009)
        e4_cat = torch.cat([e4, d_emb], dim=1)  # (N, 544, H/32, W/32)

        # --- Decoder ---
        d4 = self.decoder4(e4_cat, e3)  # (N, 256, H/16, W/16)
        d3 = self.decoder3(d4, e2)  # (N, 128, H/8, W/8)
        d2 = self.decoder2(d3, e1)  # (N, 64, H/4, W/4)
        d1 = self.decoder1(d2, e0)  # (N, 64, H/2, W/2)

        # --- Final Head ---
        logits = self.final_conv(d1)  # (N, 1, H, W)

        return logits, None
