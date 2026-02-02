import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DilatedDenseBlock(nn.Module):
    """
    A single dense block with dilated convolution.
    Structure: BN -> ReLU -> Dilated Conv1d -> Dropout
    Output: Concatenation of Input and Block Output
    """

    def __init__(self, in_channels, growth_rate, dilation, dropout_rate):
        super(DilatedDenseBlock, self).__init__()
        self.bn = nn.BatchNorm1d(in_channels)
        # Padding calculation for 'same' output length with kernel_size=3
        # padding = dilation * (kernel_size - 1) / 2
        # With k=3, padding = dilation
        padding = dilation
        self.conv = nn.Conv1d(
            in_channels,
            growth_rate,
            kernel_size=Config.KERNEL_SIZE,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        out = self.bn(x)
        out = F.relu(out)
        out = self.conv(out)
        out = self.dropout(out)
        return torch.cat([x, out], dim=1)


class CascadedDenseNet(nn.Module):
    """
    Cascaded Latent-Refined Dense Network.

    Stage 1: Extracts local sequence motifs using exponentially dilated dense blocks.
    Inter-Stage: Creates 'Pair-State' features by gathering partner features based on secondary structure.
    Stage 2: Models stacking thermodynamics using reset dilations on the pair features.
    Head: BiGRU + Linear projection.
    """

    def __init__(self):
        super(CascadedDenseNet, self).__init__()

        # ==== Stage 1: Local Motif Extraction ====
        # Stem: Project input features to initial hidden dimension
        self.stem = nn.Conv1d(Config.INPUT_CHANNELS, Config.GROWTH_RATE, kernel_size=1)

        self.stage1_blocks = nn.ModuleList()
        current_channels = Config.GROWTH_RATE

        # Stack dilated dense blocks
        for d in Config.STAGE1_DILATION_SCHEDULE:
            block = DilatedDenseBlock(
                in_channels=current_channels,
                growth_rate=Config.GROWTH_RATE,
                dilation=d,
                dropout_rate=Config.DROPOUT,
            )
            self.stage1_blocks.append(block)
            current_channels += Config.GROWTH_RATE

        self.stage1_out_channels = current_channels

        # ==== Inter-Stage: Latent Refinement ====
        # Compress features before gathering to manage complexity
        self.inter_compress = nn.Conv1d(
            self.stage1_out_channels, Config.INTER_STAGE_DIM, kernel_size=1
        )

        # ==== Global Aggregation ====
        # Input to GRU is the fused features (Compressed Self + Compressed Partner)
        # Cite Lesson 00031: Dynamic gathering of latent features
        gru_input_dim = Config.INTER_STAGE_DIM * 2

        # Cite Lesson 00003: Hybrid Architecture (CNN + BiGRU)
        # Cite Lesson 00004: Hidden size = Input // 2 (approx, here we use Config)
        self.gru = nn.GRU(
            input_size=gru_input_dim,
            hidden_size=Config.GRU_HIDDEN_SIZE,
            batch_first=True,
            bidirectional=True,
        )

        # ==== Output Head ====
        self.head = nn.Linear(Config.GRU_HIDDEN_SIZE * 2, Config.NUM_TARGETS)

    def forward(self, x, partner_indices):
        """
        Args:
            x: (Batch, Seq_Len, Channels)
            partner_indices: (Batch, Seq_Len) - Indices of paired bases
        """
        # Permute for Conv1d: (B, L, C) -> (B, C, L)
        x = x.permute(0, 2, 1)

        # --- Stage 1 ---
        out = self.stem(x)
        for block in self.stage1_blocks:
            out = block(out)

        # --- Inter-Stage Fusion ---
        # 1. Compress
        compressed = self.inter_compress(out)  # (B, InterDim, L)

        # 2. Dynamic Gather
        # We need to gather features from 'compressed' at indices specified by 'partner_indices'.
        # partner_indices is (B, L). compressed is (B, C, L).
        # We expand partner_indices to (B, C, L) to gather along the length dimension.

        batch_size, channels, seq_len = compressed.shape

        # Expand indices to match channel dimension
        # (B, L) -> (B, 1, L) -> (B, C, L)
        idx_expanded = partner_indices.unsqueeze(1).expand(-1, channels, -1)

        # Gather partner features
        # dim=2 corresponds to the Sequence Length dimension
        partner_features = torch.gather(compressed, 2, idx_expanded)

        # 3. Concatenate (Self + Partner)
        # Result: (B, 2*InterDim, L)
        fused = torch.cat([compressed, partner_features], dim=1)

        # --- Global Aggregation ---
        # GRU expects (B, L, C)
        # Skip Stage 2 to avoid over-parameterization (Cite Lesson 00017)
        out_gru_input = fused.permute(0, 2, 1)
        gru_out, _ = self.gru(out_gru_input)

        # --- Head ---
        logits = self.head(gru_out)

        return logits
