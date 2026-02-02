import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from library.config import Config
from library.layers import BlurPool


class DecoderBlock(nn.Module):
    """
    Standard U-Net Decoder Block: Upsample -> Concat -> ConvBlock.
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        # Input channels = upsampled channels + skip connection channels
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

    def forward(self, x, skip):
        x = self.upsample(x)

        # Handle potential slight shape mismatches (e.g. odd dimensions)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(
                x, size=skip.shape[2:], mode="bilinear", align_corners=True
            )

        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class AntiAliasedResNetUNet(nn.Module):
    """
    Anti-Aliased ResNet18 U-Net for Multi-Task Chest X-Ray Analysis.

    Backbone: ResNet18 with BlurPool downsampling.
    Decoder: U-Net style with skip connections.
    Heads:
        1. Classification (Study Level): GAP -> Linear
        2. Segmentation (Image Level): Conv 1x1
    """

    def __init__(self):
        super().__init__()

        # 1. Load Backbone
        weights = "IMAGENET1K_V1" if Config.PRETRAINED else None
        backbone = models.resnet18(weights=weights)

        # 2. Apply Anti-Aliasing (BlurPool)
        if Config.USE_ANTI_ALIASING:
            self._replace_layers_with_blurpool(backbone)

        # 3. Extract Encoder Layers
        # Input: (B, 3, H, W) -> Conv1: (B, 64, H/2, W/2)
        self.encoder_conv1 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        # MaxPool: (B, 64, H/4, W/4)
        self.encoder_maxpool = backbone.maxpool
        # Layer1: (B, 64, H/4, W/4) (No downsampling in ResNet18 layer1)
        self.encoder_layer1 = backbone.layer1
        # Layer2: (B, 128, H/8, W/8)
        self.encoder_layer2 = backbone.layer2
        # Layer3: (B, 256, H/16, W/16)
        self.encoder_layer3 = backbone.layer3
        # Layer4: (B, 512, H/32, W/32)
        self.encoder_layer4 = backbone.layer4

        # 4. Decoder Construction
        # Bottleneck is encoder_layer4 output (512 channels)

        # Decoder 4: Upsample 512 -> 256, Cat with Layer3 (256) -> Out 256
        self.dec4 = DecoderBlock(in_channels=512, skip_channels=256, out_channels=256)

        # Decoder 3: Upsample 256 -> 128, Cat with Layer2 (128) -> Out 128
        self.dec3 = DecoderBlock(in_channels=256, skip_channels=128, out_channels=128)

        # Decoder 2: Upsample 128 -> 64, Cat with Layer1 (64) -> Out 64
        self.dec2 = DecoderBlock(in_channels=128, skip_channels=64, out_channels=64)

        # Decoder 1: Upsample 64 -> 64, Cat with Conv1 output (64) -> Out 64
        # Note: Conv1 output is H/2, Layer1 is H/4. Decoder 2 output is H/4.
        # This block upsamples to H/2.
        self.dec1 = DecoderBlock(in_channels=64, skip_channels=64, out_channels=64)

        # Final Upsample Block to restore original resolution (H, W)
        self.final_upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # 5. Heads

        # Classification Head (Study Level)
        # Attached to the deepest feature map (Layer4)
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, Config.NUM_CLASSES_STUDY),
        )

        # Segmentation Head (Image Level)
        # Attached to the final decoder output
        self.seg_head = nn.Conv2d(32, Config.NUM_CLASSES_IMAGE, kernel_size=1)

    def _replace_layers_with_blurpool(self, model):
        """
        Replaces standard MaxPool and strided Convolutions with Anti-Aliased versions.
        """
        # A. Replace MaxPool
        # Standard: MaxPool2d(k=3, s=2, p=1)
        # New: MaxPool2d(k=3, s=1, p=1) -> BlurPool(s=2)
        model.maxpool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BlurPool(channels=64, stride=2),
        )

        # B. Replace Strided Convolutions in BasicBlocks (Layer 2, 3, 4)
        for layer_name in ["layer2", "layer3", "layer4"]:
            layer = getattr(model, layer_name)
            # In ResNet18, the first block of these layers performs downsampling
            block = layer[0]

            # 1. Modify Main Conv Path
            # Original: Conv2d(..., stride=2)
            # New: Conv2d(..., stride=1) -> BlurPool(stride=2)
            if hasattr(block, "conv1") and block.conv1.stride == (2, 2):
                old_conv = block.conv1
                block.conv1 = nn.Sequential(
                    nn.Conv2d(
                        old_conv.in_channels,
                        old_conv.out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False,
                    ),
                    BlurPool(channels=old_conv.out_channels, stride=2),
                )

            # 2. Modify Downsample Shortcut Path
            # Original: Sequential(Conv2d(..., stride=2), BN)
            # New: Sequential(Conv2d(..., stride=1), BlurPool(stride=2), BN)
            if block.downsample is not None:
                # Assuming downsample is Sequential(Conv, BN)
                old_down_conv = block.downsample[0]
                old_bn = block.downsample[1]

                if isinstance(old_down_conv, nn.Conv2d) and old_down_conv.stride == (
                    2,
                    2,
                ):
                    new_down_conv = nn.Sequential(
                        nn.Conv2d(
                            old_down_conv.in_channels,
                            old_down_conv.out_channels,
                            kernel_size=1,
                            stride=1,
                            bias=False,
                        ),
                        BlurPool(channels=old_down_conv.out_channels, stride=2),
                        old_bn,  # Re-use BN
                    )
                    block.downsample = new_down_conv

    def forward(self, x):
        # --- Encoder ---
        # x: (B, 3, 512, 512)

        x0 = self.encoder_conv1(x)  # (B, 64, 256, 256)
        x1 = self.encoder_maxpool(x0)  # (B, 64, 128, 128)
        x2 = self.encoder_layer1(x1)  # (B, 64, 128, 128)
        x3 = self.encoder_layer2(x2)  # (B, 128, 64, 64)
        x4 = self.encoder_layer3(x3)  # (B, 256, 32, 32)
        x5 = self.encoder_layer4(x4)  # (B, 512, 16, 16)

        # --- Classification Head ---
        cls_logits = self.cls_head(x5)

        # --- Decoder ---
        d4 = self.dec4(x5, x4)  # -> (256, 32, 32)
        d3 = self.dec3(d4, x3)  # -> (128, 64, 64)
        d2 = self.dec2(d3, x2)  # -> (64, 128, 128)
        d1 = self.dec1(d2, x0)  # -> (64, 256, 256)

        # --- Final Upsample & Segmentation Head ---
        f = self.final_upsample(d1)  # -> (64, 512, 512)
        f = self.final_conv(f)  # -> (32, 512, 512)
        seg_logits = self.seg_head(f)  # -> (1, 512, 512)

        return cls_logits, seg_logits
