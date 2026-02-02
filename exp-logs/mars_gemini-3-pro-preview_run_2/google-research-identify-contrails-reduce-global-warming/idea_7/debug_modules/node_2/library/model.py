import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from library.layers import DeformableConv2d, SCSEModule, ASPP


class DecoderBlock(nn.Module):
    """
    U-Net Decoder Block with SCSE Attention.
    Performs Bilinear Upsampling -> Concatenation -> Conv -> BN -> ReLU -> Conv -> BN -> ReLU -> SCSE.
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()

        # The input to the first conv is the concatenation of upsampled input and skip connection
        self.conv1 = nn.Conv2d(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.attention = SCSEModule(out_channels)

    def forward(self, x, skip=None):
        # Bilinear Upsampling
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

        if skip is not None:
            # Handle slight dimension mismatches due to rounding in pooling layers
            if x.size() != skip.size():
                x = F.interpolate(
                    x, size=skip.shape[2:], mode="bilinear", align_corners=False
                )
            x = torch.cat([x, skip], dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        # Apply Attention
        x = self.attention(x)
        return x


class DeformableResNetUNet(nn.Module):
    """
    U-Net with Deformable ResNet18 Encoder.

    - Encoder: ResNet18 with Deformable Convolutions in Layer 3 and Layer 4.
    - Bridge: ASPP (Atrous Spatial Pyramid Pooling).
    - Decoder: U-Net style with SCSE Attention.
    - Input: 6 Channels (Ash + Temporal).
    """

    def __init__(self, n_channels=6, n_classes=1, pretrained=True):
        super(DeformableResNetUNet, self).__init__()

        # 1. Load Backbone
        # Using IMAGENET1K_V1 weights if pretrained is True
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.encoder = models.resnet18(weights=weights)

        # 2. Modify Input Layer
        # ResNet18 expects 3 channels. We have n_channels (6).
        # We replace the first conv layer.
        self.encoder.conv1 = nn.Conv2d(
            n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        # Initialize the new layer using Kaiming init (since input stats differ from RGB)
        nn.init.kaiming_normal_(
            self.encoder.conv1.weight, mode="fan_out", nonlinearity="relu"
        )

        # 3. Inject Deformable Convolutions
        # Replace 3x3 convolutions in layer3 and layer4 with DeformableConv2d
        self._replace_with_deformable(self.encoder.layer3)
        self._replace_with_deformable(self.encoder.layer4)

        # 4. Define Bridge (ASPP)
        # ResNet18 Layer 4 output channels: 512
        self.bridge = ASPP(in_channels=512, out_channels=256)

        # 5. Define Decoder
        # Layer channels for ResNet18:
        # layer4: 512, layer3: 256, layer2: 128, layer1: 64, conv1: 64

        # Decoder 4: Upsamples Bridge (256) + Skips Layer 3 (256) -> Out 256
        self.decoder4 = DecoderBlock(
            in_channels=256, skip_channels=256, out_channels=256
        )

        # Decoder 3: Upsamples Dec4 (256) + Skips Layer 2 (128) -> Out 128
        self.decoder3 = DecoderBlock(
            in_channels=256, skip_channels=128, out_channels=128
        )

        # Decoder 2: Upsamples Dec3 (128) + Skips Layer 1 (64) -> Out 64
        self.decoder2 = DecoderBlock(in_channels=128, skip_channels=64, out_channels=64)

        # Decoder 1: Upsamples Dec2 (64) + Skips Conv1 (64) -> Out 64
        self.decoder1 = DecoderBlock(in_channels=64, skip_channels=64, out_channels=64)

        # 6. Final Head
        # Upsample Dec1 (64) to original resolution -> Conv 1x1
        self.final_upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )
        self.final_conv = nn.Conv2d(64, n_classes, kernel_size=1)

    def _replace_with_deformable(self, module):
        """
        Recursively replaces standard 3x3 Conv2d layers with DeformableConv2d.
        Copies weights from the original convolution to the deformable one.
        """
        for name, child in module.named_children():
            if isinstance(child, nn.Conv2d) and child.kernel_size == (3, 3):
                # Create replacement layer
                new_layer = DeformableConv2d(
                    in_channels=child.in_channels,
                    out_channels=child.out_channels,
                    kernel_size=child.kernel_size[0],
                    stride=child.stride,
                    padding=child.padding,
                    dilation=child.dilation,
                    groups=child.groups,
                    bias=(child.bias is not None),
                )

                # Copy weights
                new_layer.weight.data = child.weight.data.clone()
                if child.bias is not None:
                    new_layer.bias.data = child.bias.data.clone()

                # Replace in parent module
                setattr(module, name, new_layer)
            else:
                # Recurse
                self._replace_with_deformable(child)

    def forward(self, x):
        # --- Encoder ---
        # Input: [B, 6, H, W]
        x0 = self.encoder.conv1(x)  # [B, 64, H/2, W/2]
        x0 = self.encoder.bn1(x0)
        x0 = self.encoder.relu(x0)

        x1 = self.encoder.maxpool(x0)  # [B, 64, H/4, W/4]
        x1 = self.encoder.layer1(x1)  # [B, 64, H/4, W/4]

        x2 = self.encoder.layer2(x1)  # [B, 128, H/8, W/8]
        x3 = self.encoder.layer3(x2)  # [B, 256, H/16, W/16] (Deformable)
        x4 = self.encoder.layer4(x3)  # [B, 512, H/32, W/32] (Deformable)

        # --- Bridge ---
        b = self.bridge(x4)  # [B, 256, H/32, W/32]

        # --- Decoder ---
        d4 = self.decoder4(b, x3)  # [B, 256, H/16, W/16]
        d3 = self.decoder3(d4, x2)  # [B, 128, H/8, W/8]
        d2 = self.decoder2(d3, x1)  # [B, 64, H/4, W/4]
        d1 = self.decoder1(d2, x0)  # [B, 64, H/2, W/2]

        # --- Head ---
        out = self.final_upsample(d1)  # [B, 64, H, W]
        out = self.final_conv(out)  # [B, 1, H, W]

        return out
