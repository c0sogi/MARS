import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config
from library.layers import ConvNeXtBlock, SCSEModule, ASPP


class DecoderBlock(nn.Module):
    """
    A decoder block that performs upsampling, feature fusion, and large-kernel refinement.

    Structure:
    1. Bilinear Upsampling
    2. Concatenation with Skip Connection (optional)
    3. SCSE Attention (Spatial & Channel Squeeze-and-Excitation)
    4. 1x1 Projection (Channel Reduction)
    5. ConvNeXt Block (7x7 Depthwise Conv for refinement)
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )

        # Calculate total channels after concatenation
        concat_channels = in_channels + skip_channels

        # Attention mechanism to suppress noise in the concatenated features
        self.scse = SCSEModule(concat_channels)

        # Project concatenated features to the target dimension
        self.proj = nn.Sequential(
            nn.Conv2d(concat_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Large-kernel refinement using ConvNeXt block
        self.block = ConvNeXtBlock(dim=out_channels)

    def forward(self, x, skip=None):
        # Upsample
        x = self.upsample(x)

        # Concatenate skip connection
        if skip is not None:
            # Handle slight dimension mismatches due to padding/interpolation
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(
                    x, size=skip.shape[2:], mode="bilinear", align_corners=False
                )
            x = torch.cat([x, skip], dim=1)

        # Apply Attention
        x = self.scse(x)

        # Project and Refine
        x = self.proj(x)
        x = self.block(x)
        return x


class ConvNeXtUNet(nn.Module):
    """
    Large-Kernel ConvNeXt U-Net for Contrail Segmentation.

    Encoder: ConvNeXt-Tiny (Pretrained, 6-channel input adapted)
    Bottleneck: ASPP
    Decoder: Custom Large-Kernel Decoder Blocks with SCSE
    """

    def __init__(self):
        super().__init__()

        # 1. Encoder Setup
        # Load pretrained ConvNeXt-Tiny, adapted for 6 input channels
        self.encoder = timm.create_model(
            Config.BACKBONE,
            pretrained=True,
            features_only=True,
            in_chans=Config.IN_CHANNELS,
        )

        # Retrieve feature channel counts from the encoder
        # Typically [96, 192, 384, 768] for ConvNeXt-Tiny
        enc_channels = self.encoder.feature_info.channels()

        # 2. Bottleneck
        # ASPP applied to the deepest feature map (Stride 32)
        self.aspp = ASPP(
            in_channels=enc_channels[-1], out_channels=Config.DECODER_CHANNELS[0]
        )

        # 3. Decoder Setup
        # Config.DECODER_CHANNELS = [256, 128, 64, 32, 16]
        decoder_dims = Config.DECODER_CHANNELS

        # Block 1: Stride 32 -> 16. Skip from Encoder Stage 3 (Stride 16)
        self.decoder1 = DecoderBlock(
            in_channels=decoder_dims[0],
            skip_channels=enc_channels[2],
            out_channels=decoder_dims[0],
        )

        # Block 2: Stride 16 -> 8. Skip from Encoder Stage 2 (Stride 8)
        self.decoder2 = DecoderBlock(
            in_channels=decoder_dims[0],
            skip_channels=enc_channels[1],
            out_channels=decoder_dims[1],
        )

        # Block 3: Stride 8 -> 4. Skip from Encoder Stage 1 (Stride 4)
        self.decoder3 = DecoderBlock(
            in_channels=decoder_dims[1],
            skip_channels=enc_channels[0],
            out_channels=decoder_dims[2],
        )

        # Block 4: Stride 4 -> 2. No Skip (Stem is stride 4)
        self.decoder4 = DecoderBlock(
            in_channels=decoder_dims[2], skip_channels=0, out_channels=decoder_dims[3]
        )

        # Block 5: Stride 2 -> 1. No Skip.
        self.decoder5 = DecoderBlock(
            in_channels=decoder_dims[3], skip_channels=0, out_channels=decoder_dims[4]
        )

        # 4. Final Classification Head
        self.final_conv = nn.Conv2d(decoder_dims[4], 1, kernel_size=1)

    def forward(self, x):
        # --- Encoder ---
        # features list: [s4, s8, s16, s32]
        features = self.encoder(x)

        # --- Bottleneck ---
        # Process the deepest features (s32)
        x = self.aspp(features[3])

        # --- Decoder ---
        # Step 1: s32 -> s16 (Fuse with s16 features)
        x = self.decoder1(x, features[2])

        # Step 2: s16 -> s8 (Fuse with s8 features)
        x = self.decoder2(x, features[1])

        # Step 3: s8 -> s4 (Fuse with s4 features)
        x = self.decoder3(x, features[0])

        # Step 4: s4 -> s2 (Upsample only)
        x = self.decoder4(x)

        # Step 5: s2 -> s1 (Upsample only)
        x = self.decoder5(x)

        # --- Head ---
        logits = self.final_conv(x)

        return logits
