import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config
from library.modules import ResidualBlock1D, ASPP, AttentionGate


class DecimatedDeepSupervisionHead(nn.Module):
    """
    Auxiliary output head for Deep Supervision at lower temporal resolutions.
    Projects feature maps to the target dimension (e.g., 2 for DeltaNorth/DeltaEast).
    The 'Decimated' aspect refers to the target matching strategy during loss calculation,
    while this module performs the projection of the downsampled feature map.
    """

    def __init__(self, in_channels, out_channels):
        super(DecimatedDeepSupervisionHead, self).__init__()
        self.head = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.head(x)


class PhaseAwareAttentionResUNet(nn.Module):
    """
    Phase-Aware Stratified 1D ResUNet with Attention Gates.

    This architecture is designed to process stratified GNSS sensor data.
    It uses a Residual Encoder-Decoder structure with an ASPP bottleneck.
    Attention Gates are applied to skip connections to selectively suppress
    noisy features (e.g., from high-risk strata) based on the global context
    provided by the decoder.
    """

    def __init__(self, config=None):
        super(PhaseAwareAttentionResUNet, self).__init__()

        if config is None:
            self.config = Config()
        else:
            self.config = config

        # =====================================================================
        # Encoder
        # =====================================================================
        self.encoder_blocks = nn.ModuleList()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        in_ch = self.config.INPUT_CHANNELS

        for out_ch in self.config.ENCODER_FILTERS:
            self.encoder_blocks.append(ResidualBlock1D(in_ch, out_ch))
            in_ch = out_ch

        # =====================================================================
        # Bottleneck (ASPP)
        # =====================================================================
        # Input channels = output of last encoder block
        aspp_in_ch = self.config.ENCODER_FILTERS[-1]
        # We set ASPP out to match last encoder filter count for compatibility
        self.aspp = ASPP(aspp_in_ch, aspp_in_ch, self.config.ASPP_DILATIONS)

        # =====================================================================
        # Decoder
        # =====================================================================
        self.up_convs = nn.ModuleList()
        self.att_gates = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.aux_heads = nn.ModuleList()

        # Decoder filters: e.g., [256, 128, 64, 32]
        decoder_filters = self.config.DECODER_FILTERS
        # Skip connection channels correspond to encoder outputs in reverse order
        skip_channels = self.config.ENCODER_FILTERS[::-1]

        # Current channels starting from bottleneck output
        curr_ch = aspp_in_ch

        for i, out_ch in enumerate(decoder_filters):
            # 1. Upsampling Layer
            self.up_convs.append(
                nn.ConvTranspose1d(curr_ch, out_ch, kernel_size=2, stride=2)
            )

            # 2. Attention Gate
            # Gating signal (g) comes from the upsampled decoder feature (channels=out_ch)
            # Skip connection (x) comes from the encoder (channels=skip_ch)
            skip_ch = skip_channels[i]

            if self.config.USE_ATTENTION_GATES:
                # Intermediate channels for the attention mechanism usually reduced
                self.att_gates.append(
                    AttentionGate(f_g=out_ch, f_l=skip_ch, f_int=out_ch // 2)
                )
            else:
                self.att_gates.append(nn.Identity())

            # 3. Decoder Residual Block
            # Input to the block is the concatenation of upsampled feature and gated skip
            self.decoder_blocks.append(ResidualBlock1D(out_ch + skip_ch, out_ch))

            # 4. Deep Supervision Head
            # Add aux heads for intermediate layers (excluding the final full-res layer which is handled separately)
            if self.config.USE_DEEP_SUPERVISION and i < len(decoder_filters) - 1:
                self.aux_heads.append(
                    DecimatedDeepSupervisionHead(out_ch, self.config.OUTPUT_CHANNELS)
                )
            else:
                self.aux_heads.append(None)

            curr_ch = out_ch

        # =====================================================================
        # Final Output Head
        # =====================================================================
        self.final_conv = nn.Conv1d(curr_ch, self.config.OUTPUT_CHANNELS, kernel_size=1)

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape [Batch, Channels, Length].

        Returns:
            torch.Tensor: Final output [Batch, OutputChannels, Length].
            list[torch.Tensor] (optional): Auxiliary outputs if training and deep supervision enabled.
        """
        # --- Encoder Pass ---
        skips = []
        for block in self.encoder_blocks:
            x = block(x)
            skips.append(x)
            x = self.pool(x)

        # --- Bottleneck ---
        x = self.aspp(x)

        # --- Decoder Pass ---
        # Skips are needed in reverse order (deepest first)
        skips = skips[::-1]

        aux_outputs = []

        for i in range(len(self.decoder_blocks)):
            # 1. Upsample
            x = self.up_convs[i](x)

            skip = skips[i]

            # 2. Align shapes
            # Padding in the dataset might cause slight mismatches after pooling/upsampling
            if x.shape[2] != skip.shape[2]:
                x = F.interpolate(
                    x, size=skip.shape[2], mode="linear", align_corners=True
                )

            # 3. Apply Attention Gate
            if self.config.USE_ATTENTION_GATES:
                # g=x (upsampled decoder feature), x=skip (encoder feature)
                gated_skip = self.att_gates[i](g=x, x=skip)
            else:
                gated_skip = skip

            # 4. Concatenate
            x = torch.cat([x, gated_skip], dim=1)

            # 5. Process with Decoder Block
            x = self.decoder_blocks[i](x)

            # 6. Collect Aux Output
            if self.aux_heads[i] is not None:
                aux_outputs.append(self.aux_heads[i](x))

        # --- Final Prediction ---
        final_out = self.final_conv(x)

        if self.training and self.config.USE_DEEP_SUPERVISION:
            return final_out, aux_outputs
        else:
            return final_out
