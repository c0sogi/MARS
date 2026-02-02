import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config
from library.layers import ResidualBlock1D, ASPP, AttentionGate1D


class StratifiedAttentionResUNet(nn.Module):
    """
    Stratified 1D Residual U-Net with Attention Gates and Decimated Deep Supervision.

    Architecture:
    - Encoder: 4 levels of ResBlocks + MaxPool
    - Bridge: ASPP
    - Decoder: Upsample -> Attention Gate (Skip) -> Concat -> ResBlock
    - Heads: Final head (Scale 0) + 3 Auxiliary heads (Scales 1, 2, 3)
    """

    def __init__(
        self,
        in_channels=Config.IN_CHANNELS,
        out_channels=Config.OUT_CHANNELS,
        filters=Config.ENCODER_FILTERS,
        deep_supervision=Config.DEEP_SUPERVISION,
    ):
        super().__init__()

        self.deep_supervision = deep_supervision
        self.filters = filters

        # --- Encoder ---
        # Level 0 (Full Res)
        self.enc0 = ResidualBlock1D(in_channels, filters[0])
        self.pool0 = nn.MaxPool1d(2)

        # Level 1 (1/2 Res)
        self.enc1 = ResidualBlock1D(filters[0], filters[1])
        self.pool1 = nn.MaxPool1d(2)

        # Level 2 (1/4 Res)
        self.enc2 = ResidualBlock1D(filters[1], filters[2])
        self.pool2 = nn.MaxPool1d(2)

        # Level 3 (1/8 Res)
        self.enc3 = ResidualBlock1D(filters[2], filters[3])

        # --- Bridge ---
        # ASPP at 1/8 Resolution
        self.aspp = ASPP(filters[3], filters[3], dilations=Config.ASPP_DILATIONS)

        # --- Decoder ---
        # Up 1: Level 3 -> Level 2
        self.up1 = nn.Upsample(scale_factor=2, mode="linear", align_corners=False)
        # Gate: Gating signal from Coarser (F3), Skip from Finer (F2)
        self.att1 = AttentionGate1D(
            F_g=filters[3], F_l=filters[2], F_int=filters[2] // 2
        )
        self.dec1 = ResidualBlock1D(filters[3] + filters[2], filters[2])

        # Up 2: Level 2 -> Level 1
        self.up2 = nn.Upsample(scale_factor=2, mode="linear", align_corners=False)
        self.att2 = AttentionGate1D(
            F_g=filters[2], F_l=filters[1], F_int=filters[1] // 2
        )
        self.dec2 = ResidualBlock1D(filters[2] + filters[1], filters[1])

        # Up 3: Level 1 -> Level 0
        self.up3 = nn.Upsample(scale_factor=2, mode="linear", align_corners=False)
        self.att3 = AttentionGate1D(
            F_g=filters[1], F_l=filters[0], F_int=filters[0] // 2
        )
        self.dec3 = ResidualBlock1D(filters[1] + filters[0], filters[0])

        # --- Output Heads ---
        # Final Output (Scale 0)
        self.final_head = nn.Conv1d(filters[0], out_channels, kernel_size=1)

        if self.deep_supervision:
            # Aux Head 1 (Scale 1 - 1/2 Res)
            self.aux_head1 = nn.Conv1d(filters[1], out_channels, kernel_size=1)
            # Aux Head 2 (Scale 2 - 1/4 Res)
            self.aux_head2 = nn.Conv1d(filters[2], out_channels, kernel_size=1)
            # Aux Head 3 (Scale 3 - 1/8 Res) - From Bridge
            self.aux_head3 = nn.Conv1d(filters[3], out_channels, kernel_size=1)

    def forward(self, x):
        # --- Encoder Path ---
        x0 = self.enc0(x)  # Level 0
        p0 = self.pool0(x0)  # -> Level 1

        x1 = self.enc1(p0)  # Level 1
        p1 = self.pool1(x1)  # -> Level 2

        x2 = self.enc2(p1)  # Level 2
        p2 = self.pool2(x2)  # -> Level 3

        x3 = self.enc3(p2)  # Level 3

        # --- Bridge ---
        b = self.aspp(x3)  # Level 3

        # --- Decoder Path ---

        # Block 1 (Target: Level 2)
        d1_up = self.up1(b)
        # Handle potential size mismatch (e.g. odd input length)
        if d1_up.size(2) != x2.size(2):
            d1_up = F.interpolate(
                d1_up, size=x2.size(2), mode="linear", align_corners=False
            )

        x2_gated = self.att1(g=d1_up, x=x2)
        d1_cat = torch.cat([d1_up, x2_gated], dim=1)
        d1 = self.dec1(d1_cat)

        # Block 2 (Target: Level 1)
        d2_up = self.up2(d1)
        if d2_up.size(2) != x1.size(2):
            d2_up = F.interpolate(
                d2_up, size=x1.size(2), mode="linear", align_corners=False
            )

        x1_gated = self.att2(g=d2_up, x=x1)
        d2_cat = torch.cat([d2_up, x1_gated], dim=1)
        d2 = self.dec2(d2_cat)

        # Block 3 (Target: Level 0)
        d3_up = self.up3(d2)
        if d3_up.size(2) != x0.size(2):
            d3_up = F.interpolate(
                d3_up, size=x0.size(2), mode="linear", align_corners=False
            )

        x0_gated = self.att3(g=d3_up, x=x0)
        d3_cat = torch.cat([d3_up, x0_gated], dim=1)
        d3 = self.dec3(d3_cat)

        # --- Outputs ---
        outputs = []

        # Final Resolution Output
        outputs.append(self.final_head(d3))

        if self.deep_supervision:
            # Append auxiliary outputs in order of decreasing resolution (increasing scale index)
            # Scale 1 (1/2)
            outputs.append(self.aux_head1(d2))
            # Scale 2 (1/4)
            outputs.append(self.aux_head2(d1))
            # Scale 3 (1/8)
            outputs.append(self.aux_head3(b))

        return outputs
