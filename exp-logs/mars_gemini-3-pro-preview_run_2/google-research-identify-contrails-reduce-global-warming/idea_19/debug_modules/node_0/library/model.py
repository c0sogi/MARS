import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from library.config import Config
from library.modules import LayerNorm, ConvNeXtBlock, SCSEModule, AttentionGate, ASPP


class DecoderBlock(nn.Module):
    """
    Decoder block integrating Attention Gates, Upsampling, and Isotropic ConvNeXt refinement.
    """

    def __init__(self, in_channels, skip_channels, out_channels, use_ag=True):
        super().__init__()
        self.use_ag = use_ag

        # Upsampling path: Bilinear -> Conv 1x1 -> Norm
        # We reduce channels during upsampling or keep them and reduce after concat.
        # Here we map in_channels -> out_channels during upsampling.
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            LayerNorm(out_channels, data_format="channels_first"),
        )

        # Attention Gate
        # Filters the skip connection (x) using the gating signal (g)
        if self.use_ag and skip_channels > 0:
            # F_g: Gating signal channels (from coarser layer, here in_channels)
            # F_l: Skip connection channels
            # F_int: Intermediate channels (usually smaller)
            self.ag = AttentionGate(
                F_g=in_channels, F_l=skip_channels, F_int=out_channels
            )
        else:
            self.ag = None

        # Feature Fusion and Reduction
        # Input dim = out_channels (from up) + skip_channels (from skip)
        concat_channels = out_channels + skip_channels
        self.reduce = nn.Conv2d(concat_channels, out_channels, kernel_size=1)

        # Isotropic Refinement (ConvNeXt Block)
        self.block = ConvNeXtBlock(dim=out_channels, drop_path=0.0)

        # Recalibration
        self.scse = SCSEModule(out_channels)

    def forward(self, x, skip=None):
        """
        Args:
            x: Input feature map from the deeper/coarser layer (Gating Signal).
            skip: Skip connection feature map from the encoder.
        """
        # 1. Apply Attention Gate
        # We use the coarser signal 'x' to gate the skip connection 'skip'
        if self.use_ag and skip is not None:
            skip = self.ag(g=x, x=skip)

        # 2. Upsample the main path
        x_up = self.up(x)

        # 3. Concatenate
        if skip is not None:
            # Handle potential spatial dimension mismatches (e.g., odd input sizes)
            if x_up.shape[2:] != skip.shape[2:]:
                x_up = F.interpolate(
                    x_up, size=skip.shape[2:], mode="bilinear", align_corners=False
                )
            out = torch.cat([x_up, skip], dim=1)
        else:
            out = x_up

        # 4. Reduce channels, Refine, and Recalibrate
        out = self.reduce(out)
        out = self.block(out)
        out = self.scse(out)

        return out


class FinalUpsampleBlock(nn.Module):
    """
    Final upsampling block to bridge Stride 4 (Stem) to Stride 1 (Pixel Level).
    Does not use skip connections as the backbone stem is Stride 4.
    """

    def __init__(self, in_channels, out_channels, scale_factor=4):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(
                scale_factor=scale_factor, mode="bilinear", align_corners=False
            ),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            LayerNorm(out_channels, data_format="channels_first"),
        )
        self.block = ConvNeXtBlock(dim=out_channels, drop_path=0.0)
        self.scse = SCSEModule(out_channels)

    def forward(self, x):
        x = self.up(x)
        x = self.block(x)
        x = self.scse(x)
        return x


class AttentionGatedConvNeXtUNet(nn.Module):
    def __init__(self):
        super().__init__()

        # ==========================================
        # 1. Encoder (Backbone)
        # ==========================================
        # ConvNeXt-Tiny. features_only=True gives features at strides [4, 8, 16, 32].
        # in_chans=6 adapts the first layer to accept our 6-channel input.
        self.encoder = timm.create_model(
            Config.BACKBONE,
            pretrained=True if Config.ENCODER_WEIGHTS else False,
            in_chans=Config.N_CHANNELS,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )

        # Get channel counts for each stage: [96, 192, 384, 768] for tiny
        enc_channels = self.encoder.feature_info.channels()

        # ==========================================
        # 2. Bottleneck (Context Bridge)
        # ==========================================
        # ASPP applied to the deepest feature map (Stride 32)
        # Projects to the first decoder channel dimension
        self.aspp = ASPP(
            in_channels=enc_channels[3], out_channels=Config.DECODER_CHANNELS[0]
        )

        # ==========================================
        # 3. Decoder
        # ==========================================
        # Config.DECODER_CHANNELS = [256, 128, 64, 32, 16]

        # Stage 1: Stride 32 -> 16
        # Skip: Encoder Stride 16 (enc_channels[2])
        self.dec1 = DecoderBlock(
            in_channels=Config.DECODER_CHANNELS[0],
            skip_channels=enc_channels[2],
            out_channels=Config.DECODER_CHANNELS[1],
            use_ag=Config.USE_ATTENTION_GATES,
        )

        # Stage 2: Stride 16 -> 8
        # Skip: Encoder Stride 8 (enc_channels[1])
        self.dec2 = DecoderBlock(
            in_channels=Config.DECODER_CHANNELS[1],
            skip_channels=enc_channels[1],
            out_channels=Config.DECODER_CHANNELS[2],
            use_ag=Config.USE_ATTENTION_GATES,
        )

        # Stage 3: Stride 8 -> 4
        # Skip: Encoder Stride 4 (enc_channels[0])
        self.dec3 = DecoderBlock(
            in_channels=Config.DECODER_CHANNELS[2],
            skip_channels=enc_channels[0],
            out_channels=Config.DECODER_CHANNELS[3],
            use_ag=Config.USE_ATTENTION_GATES,
        )

        # Stage 4: Stride 4 -> 1
        # No skip connection available (Backbone stem is Stride 4)
        self.dec4 = FinalUpsampleBlock(
            in_channels=Config.DECODER_CHANNELS[3],
            out_channels=Config.DECODER_CHANNELS[4],
            scale_factor=4,
        )

        # ==========================================
        # 4. Segmentation Head
        # ==========================================
        self.head = nn.Conv2d(Config.DECODER_CHANNELS[4], 1, kernel_size=1)

    def forward(self, x):
        # x shape: (Batch, 6, H, W)

        # --- Encoder ---
        enc_feats = self.encoder(x)
        # f0: stride 4
        # f1: stride 8
        # f2: stride 16
        # f3: stride 32

        # --- Bottleneck ---
        x = self.aspp(enc_feats[3])

        # --- Decoder ---
        x = self.dec1(x, enc_feats[2])  # 32 -> 16
        x = self.dec2(x, enc_feats[1])  # 16 -> 8
        x = self.dec3(x, enc_feats[0])  # 8 -> 4
        x = self.dec4(x)  # 4 -> 1

        # --- Head ---
        logits = self.head(x)

        return logits
