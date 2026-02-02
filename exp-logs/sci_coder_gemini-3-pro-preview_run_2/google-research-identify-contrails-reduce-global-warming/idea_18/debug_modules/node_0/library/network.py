import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config
from library.layers import ConvNeXtBlock, LayerNorm2d, SCSEModule, AttentionGate, ASPP


class DecoderBlock(nn.Module):
    """
    Decoder block for Attention-Gated U-Net.
    Performs: Upsampling -> Attention Gating (Optional) -> Concat -> Projection -> ConvNeXt Block -> SCSE.
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.has_skip = skip_channels > 0

        if self.has_skip:
            # Attention Gate filters skip features (x) using decoder features (g)
            # F_g = in_channels, F_l = skip_channels
            self.attention = AttentionGate(
                F_g=in_channels, F_l=skip_channels, F_int=in_channels // 2
            )
            concat_channels = in_channels + skip_channels
        else:
            concat_channels = in_channels

        # Project fused features to out_channels
        # This reduces channel dimensions before the heavy ConvNeXt block
        self.project = nn.Sequential(
            nn.Conv2d(concat_channels, out_channels, kernel_size=1, bias=False),
            LayerNorm2d(out_channels),
            nn.GELU(),
        )

        # Large Kernel Processing (7x7) to preserve linearity
        self.block = ConvNeXtBlock(dim=out_channels)

        # Feature Recalibration
        self.scse = SCSEModule(out_channels)

    def forward(self, x, skip=None):
        # 1. Bilinear Upsampling
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

        # 2. Attention Gating & Fusion
        if self.has_skip:
            if skip is None:
                raise ValueError(
                    "Skip connection expected for this block but not provided."
                )

            # Filter the skip connection
            s = self.attention(g=x, x=skip)

            # Concatenate
            x = torch.cat([x, s], dim=1)

        # 3. Projection & Processing
        x = self.project(x)
        x = self.block(x)
        x = self.scse(x)
        return x


class AttentionGatedUNet(nn.Module):
    """
    Attention-Gated Large-Kernel ConvNeXt U-Net.

    Encoder: ConvNeXt-Tiny (7x7 kernels)
    Bridge: ASPP
    Decoder: Attention Gates + ConvNeXt Blocks + SCSE
    """

    def __init__(
        self,
        encoder_name=Config.ENCODER_NAME,
        encoder_weights=Config.ENCODER_WEIGHTS,
        in_channels=Config.IN_CHANNELS,
        num_classes=Config.NUM_CLASSES,
    ):
        super().__init__()

        # ===========================
        # 1. Encoder (Backbone)
        # ===========================
        # Load ConvNeXt Tiny. features_only=True returns a list of feature maps.
        # Indices 0, 1, 2, 3 correspond to strides 4, 8, 16, 32.
        self.encoder = timm.create_model(
            encoder_name,
            pretrained=(encoder_weights == "imagenet"),
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )

        # Modify the first layer (Stem) to accept 6 channels instead of 3
        original_stem = self.encoder.stem[0]
        new_stem = nn.Conv2d(
            in_channels,
            original_stem.out_channels,
            kernel_size=original_stem.kernel_size,
            stride=original_stem.stride,
            padding=original_stem.padding,
        )

        # Smart Initialization for new channels
        with torch.no_grad():
            if original_stem.weight.shape[1] == 3:
                # Copy RGB weights to the first 3 channels
                new_stem.weight[:, :3] = original_stem.weight
                # Initialize the extra 3 channels (Temporal Diffs) with the mean of RGB weights
                # This preserves the magnitude distribution of the pretrained weights
                mean_weight = original_stem.weight.mean(dim=1, keepdim=True)
                new_stem.weight[:, 3:] = mean_weight.repeat(1, in_channels - 3, 1, 1)
            new_stem.bias = original_stem.bias

        self.encoder.stem[0] = new_stem

        # Get channel counts. For ConvNeXt-Tiny: [96, 192, 384, 768]
        enc_channels = self.encoder.feature_info.channels()

        # ===========================
        # 2. Bridge
        # ===========================
        # ASPP on the deepest feature map (Stride 32)
        self.bridge = ASPP(in_channels=enc_channels[3], out_channels=256)

        # ===========================
        # 3. Decoder
        # ===========================
        # We need to upsample from Stride 32 (Bridge) back to Stride 1 (Original Size)

        # Decoder Stage 1: Stride 32 -> 16. Fuse with Stage 2 (Stride 16)
        self.dec1 = DecoderBlock(
            in_channels=256, skip_channels=enc_channels[2], out_channels=256
        )

        # Decoder Stage 2: Stride 16 -> 8. Fuse with Stage 1 (Stride 8)
        self.dec2 = DecoderBlock(
            in_channels=256, skip_channels=enc_channels[1], out_channels=128
        )

        # Decoder Stage 3: Stride 8 -> 4. Fuse with Stage 0 (Stem, Stride 4)
        self.dec3 = DecoderBlock(
            in_channels=128, skip_channels=enc_channels[0], out_channels=64
        )

        # Decoder Stage 4: Stride 4 -> 2. No skip connection available from encoder.
        self.dec4 = DecoderBlock(in_channels=64, skip_channels=0, out_channels=32)

        # Decoder Stage 5: Stride 2 -> 1. No skip connection.
        self.dec5 = DecoderBlock(in_channels=32, skip_channels=0, out_channels=16)

        # ===========================
        # 4. Head
        # ===========================
        self.head = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x):
        # --- Encoder ---
        features = self.encoder(x)
        s0, s1, s2, s3 = features
        # s0: Stride 4 (96 ch)
        # s1: Stride 8 (192 ch)
        # s2: Stride 16 (384 ch)
        # s3: Stride 32 (768 ch)

        # --- Bridge ---
        x = self.bridge(s3)  # -> (B, 256, H/32, W/32)

        # --- Decoder ---
        x = self.dec1(x, s2)  # -> (B, 256, H/16, W/16)
        x = self.dec2(x, s1)  # -> (B, 128, H/8, W/8)
        x = self.dec3(x, s0)  # -> (B, 64, H/4, W/4)
        x = self.dec4(x)  # -> (B, 32, H/2, W/2)
        x = self.dec5(x)  # -> (B, 16, H, W)

        # --- Head ---
        logits = self.head(x)

        return logits
