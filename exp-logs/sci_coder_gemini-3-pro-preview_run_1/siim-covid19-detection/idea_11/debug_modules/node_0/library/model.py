import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class GeM(nn.Module):
    """
    Generalized Mean Pooling layer.
    Computes the generalized mean of each channel in the feature map.
    f = (1/|X| * sum(x^p))^(1/p)
    """

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # Clamp for numerical stability
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(
            1.0 / p
        )

    def __repr__(self):
        return (
            self.__class__.__name__
            + f"(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})"
        )


class PixelShuffleBlock(nn.Module):
    """
    Sub-Pixel Convolution Block for learnable upsampling.
    Consists of: Conv2d (1x1) -> PixelShuffle -> BatchNorm -> ReLU
    """

    def __init__(self, in_channels, out_channels, upscale_factor=2):
        super(PixelShuffleBlock, self).__init__()
        # To upsample by r, we need output channels * r^2
        intermediate_channels = out_channels * (upscale_factor**2)

        self.conv = nn.Conv2d(
            in_channels, intermediate_channels, kernel_size=1, bias=False
        )
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DecoderBlock(nn.Module):
    """
    U-Net Decoder Block using PixelShuffle for upsampling.
    Structure: PixelShuffleUp -> Concat(Skip) -> Conv3x3 -> Conv3x3
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()

        # Upsample input to half the channels (heuristic) or specific target
        # Here we upsample in_channels -> in_channels // 2
        self.up = PixelShuffleBlock(in_channels, in_channels // 2, upscale_factor=2)

        # Calculate channels after concatenation
        concat_channels = (in_channels // 2) + skip_channels

        # Standard Double Conv
        self.conv1 = nn.Conv2d(
            concat_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, skip=None):
        x = self.up(x)

        if skip is not None:
            # Handle slight dimension mismatches due to rounding in pooling
            if x.size(2) != skip.size(2) or x.size(3) != skip.size(3):
                x = F.interpolate(
                    x,
                    size=(skip.size(2), skip.size(3)),
                    mode="bilinear",
                    align_corners=True,
                )
            x = torch.cat([x, skip], dim=1)

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class ResNet18UNet(nn.Module):
    """
    ResNet18 U-Net with GeM Pooling and Sub-Pixel Upsampling.

    Backbone: ResNet18 (pretrained)
    Head: GeM Pooling -> Linear (Study Classification)
    Decoder: PixelShuffle U-Net (Image Segmentation)
    """

    def __init__(self, num_classes=4, pretrained=True):
        super(ResNet18UNet, self).__init__()

        # --- Encoder (ResNet18) ---
        # Load pretrained weights
        base = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )

        # Extract layers
        self.enc0 = nn.Sequential(
            base.conv1, base.bn1, base.relu
        )  # Out: 64 ch, 1/2 res (e.g. 256x256)
        self.pool = base.maxpool  # Out: 64 ch, 1/4 res (e.g. 128x128)
        self.enc1 = base.layer1  # Out: 64 ch, 1/4 res
        self.enc2 = base.layer2  # Out: 128 ch, 1/8 res
        self.enc3 = base.layer3  # Out: 256 ch, 1/16 res
        self.enc4 = base.layer4  # Out: 512 ch, 1/32 res

        # --- Classification Head (Study Level) ---
        self.gem = GeM(p=3)
        self.cls_head = nn.Linear(512, num_classes)

        # --- Decoder (Image Level) ---
        # Bottleneck Input: 512 ch (enc4)

        # Block 4: Up(512) + Skip(256 from enc3) -> Out 256
        self.dec4 = DecoderBlock(in_channels=512, skip_channels=256, out_channels=256)

        # Block 3: Up(256) + Skip(128 from enc2) -> Out 128
        self.dec3 = DecoderBlock(in_channels=256, skip_channels=128, out_channels=128)

        # Block 2: Up(128) + Skip(64 from enc1) -> Out 64
        self.dec2 = DecoderBlock(in_channels=128, skip_channels=64, out_channels=64)

        # Block 1: Up(64) + Skip(64 from enc0) -> Out 32
        # Note: enc0 is 64 channels
        self.dec1 = DecoderBlock(in_channels=64, skip_channels=64, out_channels=32)

        # Final Upsample: 32 -> 16 channels, restore to full resolution (1/2 -> 1/1)
        self.final_up = PixelShuffleBlock(
            in_channels=32, out_channels=16, upscale_factor=2
        )

        # Final Segmentation Head
        self.seg_head = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        # --- Encoder ---
        x0 = self.enc0(x)  # 1/2
        p0 = self.pool(x0)  # 1/4
        x1 = self.enc1(p0)  # 1/4
        x2 = self.enc2(x1)  # 1/8
        x3 = self.enc3(x2)  # 1/16
        x4 = self.enc4(x3)  # 1/32

        # --- Classification Branch ---
        # GeM Pooling on the deepest feature map
        pooled = self.gem(x4)
        pooled = pooled.flatten(1)
        logits = self.cls_head(pooled)

        # --- Segmentation Branch (Decoder) ---
        d4 = self.dec4(x4, x3)
        d3 = self.dec3(d4, x2)
        d2 = self.dec2(d3, x1)
        d1 = self.dec1(d2, x0)

        out = self.final_up(d1)
        mask = self.seg_head(out)

        return mask, logits
