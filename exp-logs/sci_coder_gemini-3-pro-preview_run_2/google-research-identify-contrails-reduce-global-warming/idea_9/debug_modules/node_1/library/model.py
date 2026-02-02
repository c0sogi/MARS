import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from library.config import Config


class SCSEModule(nn.Module):
    """
    Spatial and Channel Squeeze & Excitation Module.
    Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks.
    """

    def __init__(self, in_channels, reduction=16):
        super(SCSEModule, self).__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)


class DecoderBlock(nn.Module):
    """
    U-Net Decoder Block with Bilinear Upsampling and SCSE.
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()
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
        self.scse = SCSEModule(out_channels)

    def forward(self, x, skip=None):
        # Bilinear Upsampling
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)

        if skip is not None:
            # Handle potential padding issues if dimensions don't match exactly
            if x.size(2) != skip.size(2) or x.size(3) != skip.size(3):
                x = F.interpolate(
                    x,
                    size=(skip.size(2), skip.size(3)),
                    mode="bilinear",
                    align_corners=True,
                )
            x = torch.cat([x, skip], dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.scse(x)
        return x


class TransformerBottleneck(nn.Module):
    """
    Transformer Bottleneck to capture global context.
    Flattens feature map, applies Transformer Encoder, and reshapes back.
    """

    def __init__(
        self, in_channels, embed_dim, num_heads, num_layers, dropout=0.1, feature_size=8
    ):
        super(TransformerBottleneck, self).__init__()

        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.feature_size = feature_size

        # Project input channels to embedding dimension if necessary
        if in_channels != embed_dim:
            self.input_proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        else:
            self.input_proj = nn.Identity()

        # Positional Embedding
        num_patches = feature_size * feature_size
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, embed_dim))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN is generally more stable
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Project back if necessary (though usually we keep it at embed_dim for decoder)
        self.output_proj = nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape

        # Project to embed_dim
        x = self.input_proj(x)  # (B, embed_dim, H, W)

        # Flatten: (B, embed_dim, H, W) -> (B, embed_dim, H*W) -> (B, H*W, embed_dim)
        x = x.flatten(2).transpose(1, 2)

        # Add positional embedding
        x = x + self.pos_embedding

        # Transformer Pass
        x = self.transformer(x)

        # Reshape back: (B, H*W, embed_dim) -> (B, embed_dim, H*W) -> (B, embed_dim, H, W)
        x = x.transpose(1, 2).reshape(B, self.embed_dim, H, W)

        return self.output_proj(x)


class HybridResNetTransformerUNet(nn.Module):
    """
    Hybrid U-Net with ResNet18 Encoder and Transformer Bottleneck.
    """

    def __init__(self):
        super(HybridResNetTransformerUNet, self).__init__()

        # ---------------------------------------------------------
        # Encoder: ResNet18
        # ---------------------------------------------------------
        # Load pretrained weights
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Modify first conv layer to accept 6 channels
        # Original: Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        old_conv = resnet.conv1
        self.encoder_conv1 = nn.Conv2d(
            Config.INPUT_CHANNELS,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )

        # Initialize new conv weights: Copy RGB weights to both sets of 3 channels
        with torch.no_grad():
            self.encoder_conv1.weight[:, :3] = old_conv.weight
            self.encoder_conv1.weight[:, 3:] = old_conv.weight

        self.encoder_bn1 = resnet.bn1
        self.encoder_relu = resnet.relu
        self.encoder_maxpool = resnet.maxpool

        self.encoder_layer1 = resnet.layer1  # 64 channels
        self.encoder_layer2 = resnet.layer2  # 128 channels
        self.encoder_layer3 = resnet.layer3  # 256 channels
        self.encoder_layer4 = resnet.layer4  # 512 channels

        # ---------------------------------------------------------
        # Bottleneck: Transformer
        # ---------------------------------------------------------
        # ResNet18 layer4 output is 512 channels. At 256x256 input, spatial dim is 8x8.
        self.bottleneck = TransformerBottleneck(
            in_channels=512,
            embed_dim=Config.TRANSFORMER_EMBED_DIM,
            num_heads=Config.TRANSFORMER_NUM_HEADS,
            num_layers=Config.TRANSFORMER_NUM_LAYERS,
            dropout=Config.TRANSFORMER_DROPOUT,
            feature_size=Config.IMAGE_SIZE // 32,
        )

        # ---------------------------------------------------------
        # Decoder
        # ---------------------------------------------------------
        # Bottleneck out: 512
        # Skip 3 (Layer 3): 256
        self.decoder4 = DecoderBlock(
            in_channels=Config.TRANSFORMER_EMBED_DIM,
            skip_channels=256,
            out_channels=256,
        )

        # Skip 2 (Layer 2): 128
        self.decoder3 = DecoderBlock(
            in_channels=256, skip_channels=128, out_channels=128
        )

        # Skip 1 (Layer 1): 64
        self.decoder2 = DecoderBlock(in_channels=128, skip_channels=64, out_channels=64)

        # Skip 0 (Relu output): 64
        self.decoder1 = DecoderBlock(in_channels=64, skip_channels=64, out_channels=64)

        # Final Upsampling to restore original resolution
        # Current size is H/2 (128x128). Need to go to H (256x256).
        self.final_upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),  # Binary mask
        )

    def forward(self, x):
        # ---------------------------------------------------------
        # Encoder
        # ---------------------------------------------------------
        # Input: (B, 6, 256, 256)
        x = self.encoder_conv1(x)
        x = self.encoder_bn1(x)
        s0 = self.encoder_relu(x)  # Skip 0: (B, 64, 128, 128)

        x = self.encoder_maxpool(s0)
        s1 = self.encoder_layer1(x)  # Skip 1: (B, 64, 64, 64)
        s2 = self.encoder_layer2(s1)  # Skip 2: (B, 128, 32, 32)
        s3 = self.encoder_layer3(s2)  # Skip 3: (B, 256, 16, 16)
        x = self.encoder_layer4(s3)  # (B, 512, 8, 8)

        # ---------------------------------------------------------
        # Bottleneck
        # ---------------------------------------------------------
        x = self.bottleneck(x)  # (B, 512, 8, 8)

        # ---------------------------------------------------------
        # Decoder
        # ---------------------------------------------------------
        x = self.decoder4(x, s3)  # -> (B, 256, 16, 16)
        x = self.decoder3(x, s2)  # -> (B, 128, 32, 32)
        x = self.decoder2(x, s1)  # -> (B, 64, 64, 64)
        x = self.decoder1(x, s0)  # -> (B, 64, 128, 128)

        # ---------------------------------------------------------
        # Head
        # ---------------------------------------------------------
        logits = self.final_upsample(x)  # -> (B, 1, 256, 256)

        return logits
