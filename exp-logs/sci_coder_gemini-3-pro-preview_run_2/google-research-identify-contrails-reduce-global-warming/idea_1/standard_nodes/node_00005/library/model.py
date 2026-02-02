import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from library.config import Config


class ConvBlock(nn.Module):
    """
    Standard Convolutional Block: Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> ReLU
    """

    def __init__(self, in_c, out_c):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class DecoderBlock(nn.Module):
    """
    Decoder Block: Upsample -> Concat with Skip -> ConvBlock
    """

    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = ConvBlock(in_c + skip_c, out_c)

    def forward(self, x, skip):
        x = self.up(x)

        # Ensure dimensions match (handles potential rounding issues, though 256x256 is safe)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(
                x, size=skip.shape[2:], mode="bilinear", align_corners=True
            )

        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ResNet34UNet(nn.Module):
    """
    U-Net with ResNet34 Encoder for Contrail Segmentation.
    Cite solution_lesson_node_00001: Uses physics-informed 6-channel input.
    Cite solution_lesson_node_00002: Increased capacity from ResNet18 to ResNet34.

    Input: 6 Channels (Ash t=4, Difference t=4-t=3)
    Output: 1 Channel (Logits)
    """

    def __init__(self, in_channels=Config.IN_CHANNELS, num_classes=1):
        super().__init__()

        # Load Pretrained Backbone
        self.base_model = models.resnet34(weights="IMAGENET1K_V1")

        # ----------------------------------------------------------------
        # 1. Modify Input Layer
        # ----------------------------------------------------------------
        original_layer = self.base_model.conv1
        self.base_model.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=original_layer.out_channels,
            kernel_size=original_layer.kernel_size,
            stride=original_layer.stride,
            padding=original_layer.padding,
            bias=original_layer.bias is not None,
        )

        # Initialize weights for the new layer
        with torch.no_grad():
            self.base_model.conv1.weight[:, :3] = original_layer.weight
            if in_channels > 3:
                self.base_model.conv1.weight[:, 3:6] = original_layer.weight

        self.encoder_layers = list(self.base_model.children())

        # ----------------------------------------------------------------
        # 2. Define Encoder Stages
        # ----------------------------------------------------------------
        # ResNet18 structure:
        # layer0: conv1, bn1, relu (stride 2) -> 1/2, 64ch
        # layer1: maxpool (stride 2) + layer1 (stride 1) -> 1/4, 64ch
        # layer2: layer2 (stride 2) -> 1/8, 128ch
        # layer3: layer3 (stride 2) -> 1/16, 256ch
        # layer4: layer4 (stride 2) -> 1/32, 512ch

        self.enc0 = nn.Sequential(*self.encoder_layers[:3])
        self.enc1 = nn.Sequential(*self.encoder_layers[3:5])
        self.enc2 = self.encoder_layers[5]
        self.enc3 = self.encoder_layers[6]
        self.enc4 = self.encoder_layers[7]

        # ----------------------------------------------------------------
        # 3. Define Decoder Stages
        # ----------------------------------------------------------------
        # Decoder 4: Up(1/32) + Skip(1/16) -> 1/16
        # In: 512 (enc4) + 256 (enc3)
        self.dec4 = DecoderBlock(in_c=512, skip_c=256, out_c=256)

        # Decoder 3: Up(1/16) + Skip(1/8) -> 1/8
        # In: 256 (dec4) + 128 (enc2)
        self.dec3 = DecoderBlock(in_c=256, skip_c=128, out_c=128)

        # Decoder 2: Up(1/8) + Skip(1/4) -> 1/4
        # In: 128 (dec3) + 64 (enc1)
        self.dec2 = DecoderBlock(in_c=128, skip_c=64, out_c=64)

        # Decoder 1: Up(1/4) + Skip(1/2) -> 1/2
        # In: 64 (dec2) + 64 (enc0)
        self.dec1 = DecoderBlock(in_c=64, skip_c=64, out_c=32)

        # ----------------------------------------------------------------
        # 4. Final Head
        # ----------------------------------------------------------------
        # Upsample 1/2 -> 1/1
        self.final_up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        # --- Encoder ---
        x0 = self.enc0(x)  # 1/2, 64ch
        x1 = self.enc1(x0)  # 1/4, 64ch
        x2 = self.enc2(x1)  # 1/8, 128ch
        x3 = self.enc3(x2)  # 1/16, 256ch
        x4 = self.enc4(x3)  # 1/32, 512ch

        # --- Decoder ---
        d4 = self.dec4(x4, x3)  # -> 1/16
        d3 = self.dec3(d4, x2)  # -> 1/8
        d2 = self.dec2(d3, x1)  # -> 1/4
        d1 = self.dec1(d2, x0)  # -> 1/2

        # --- Head ---
        out = self.final_up(d1)  # -> 1/1
        out = self.final_conv(out)

        return out
