import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from library.config import Config


class ConvBlock2d(nn.Module):
    """
    Basic 2D Convolution Block: Conv2d -> BN -> ReLU
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ConvBlock2d, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DecoderBlock2d(nn.Module):
    """
    2D Decoder Block with Skip Connection.
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock2d, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv1 = ConvBlock2d(in_channels + skip_channels, out_channels)
        self.conv2 = ConvBlock2d(out_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)

        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(
                x, size=skip.shape[2:], mode="bilinear", align_corners=False
            )

        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class ResNetUNet2D(nn.Module):
    """
    2D U-Net with ResNet-18 Backbone.
    Cite solution_lesson_node_00012: Prefer ResNet backbones for speed.
    """

    def __init__(
        self, in_channels=Config.IN_CHANNELS, out_channels=Config.OUT_CHANNELS
    ):
        super(ResNetUNet2D, self).__init__()

        # --- Encoder (ResNet-18) ---
        weights = ResNet18_Weights.DEFAULT
        self.backbone = resnet18(weights=weights)

        # Modify first layer if in_channels != 3
        # In our case, in_channels=3 (2.5D), so we can keep it or re-init if we want specific behavior.
        # However, standard ResNet expects RGB. 2.5D is (z-1, z, z+1).
        # We keep the pre-trained weights.
        if in_channels != 3:
            original_conv = self.backbone.conv1
            self.backbone.conv1 = nn.Conv2d(
                in_channels,
                original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=False,
            )
            # Average weights if needed, but for 3->3 we just use ImageNet weights

        self.encoder_stem = nn.Sequential(
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.relu,
            self.backbone.maxpool,
        )
        self.encoder_layer1 = self.backbone.layer1  # 64
        self.encoder_layer2 = self.backbone.layer2  # 128
        self.encoder_layer3 = self.backbone.layer3  # 256
        self.encoder_layer4 = self.backbone.layer4  # 512

        # --- Decoder ---
        self.dec4 = DecoderBlock2d(512, 256, 256)
        self.dec3 = DecoderBlock2d(256, 128, 128)
        self.dec2 = DecoderBlock2d(128, 64, 64)

        # Final upsampling to match input resolution
        # ResNet stem reduces by 4 (stride 2 conv + stride 2 pool)
        # Layer 1 output is same size as stem output (1/4)
        # So dec2 output is 1/4 size. We need 2 more upsamples.

        self.dec1 = DecoderBlock2d(
            64, 64, 32
        )  # Skip connection from stem? Stem is 64ch.
        # Actually, let's use the features before maxpool as skip?
        # self.backbone.relu output is 64ch, size 1/2.

        # Revised Decoder:
        # L4 (1/32) -> L3 (1/16)
        # L3 (1/16) -> L2 (1/8)
        # L2 (1/8) -> L1 (1/4)
        # L1 (1/4) -> Stem (1/2) ?

        # Let's simplify.
        # dec4: In 512, Skip 256 -> Out 256 (Size 1/16)
        # dec3: In 256, Skip 128 -> Out 128 (Size 1/8)
        # dec2: In 128, Skip 64  -> Out 64  (Size 1/4)

        # Now we are at 1/4. We need to get to 1/1.
        # We can upsample x4 or use more blocks.
        # Let's use a final block.
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
            ConvBlock2d(64, 32),
            nn.Conv2d(32, out_channels, kernel_size=1),
        )

    def forward(self, x):
        # x: (B, 3, H, W)

        # Encoder
        x_stem = self.backbone.conv1(x)
        x_stem = self.backbone.bn1(x_stem)
        x_stem = self.backbone.relu(x_stem)  # (B, 64, H/2, W/2)
        x_pool = self.backbone.maxpool(x_stem)  # (B, 64, H/4, W/4)

        x_l1 = self.encoder_layer1(x_pool)  # (B, 64, H/4, W/4)
        x_l2 = self.encoder_layer2(x_l1)  # (B, 128, H/8, W/8)
        x_l3 = self.encoder_layer3(x_l2)  # (B, 256, H/16, W/16)
        x_l4 = self.encoder_layer4(x_l3)  # (B, 512, H/32, W/32)

        # Decoder
        d4 = self.dec4(x_l4, x_l3)
        d3 = self.dec3(d4, x_l2)
        d2 = self.dec2(d3, x_l1)

        # Final
        logits = self.final_up(d2)

        return logits
