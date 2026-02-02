import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DilatedResidualBlock(nn.Module):
    """
    A single block for the Residual Dilated Backbone (TCN style).
    Performs Dilated Conv1d -> ReLU -> Dropout -> Residual Add.
    Keeps channel dimension constant.
    """

    def __init__(self, channels, dilation, dropout):
        super(DilatedResidualBlock, self).__init__()
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=Config.KERNEL_SIZE,
            padding=dilation,
            dilation=dilation,
        )
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.conv(x)
        out = self.act(out)
        out = self.dropout(out)
        return x + out


class ResidualRefinedNet(nn.Module):
    """
    Residual-Refined Hybrid Network (Idea 8 Replication).

    Architecture:
    1. Input Projection (14 -> 64)
    2. Stage 1: Residual Dilated Backbone (Local Motif Extraction)
    3. Inter-Stage: Compression (64 -> 32) & Dynamic Pair-State Gathering
    4. Stage 2: Residual Dilated Backbone (Stacking Interaction Solver)
    5. Global Aggregation: Bidirectional GRU
    6. Head: Linear Projection
    """

    def __init__(self):
        super(ResidualRefinedNet, self).__init__()

        # Input Projection
        self.input_proj = nn.Conv1d(
            Config.INPUT_CHANNELS, Config.NUM_CHANNELS, kernel_size=1
        )

        # --- Stage 1: Local Motif Extraction ---
        self.stage1_blocks = nn.ModuleList()
        for dilation in Config.STAGE1_DILATIONS:
            block = DilatedResidualBlock(
                channels=Config.NUM_CHANNELS,
                dilation=dilation,
                dropout=Config.DROPOUT,
            )
            self.stage1_blocks.append(block)

        # --- Inter-Stage: Compression ---
        # Compresses features (64) to a compact latent state (32) for gathering
        self.compress = nn.Conv1d(
            Config.NUM_CHANNELS, Config.BOTTLENECK_DIM, kernel_size=1
        )

        # --- Stage 2: Stacking Interaction Solver ---
        # Input dimension is BOTTLENECK_DIM * 2 (32 + 32 = 64)
        # This matches NUM_CHANNELS (64), so we can use the same block width
        self.stage2_blocks = nn.ModuleList()
        current_dim = Config.BOTTLENECK_DIM * 2

        # Ensure Stage 2 width matches NUM_CHANNELS
        assert (
            current_dim == Config.NUM_CHANNELS
        ), "Stage 2 input dim must match NUM_CHANNELS"

        for dilation in Config.STAGE2_DILATIONS:
            block = DilatedResidualBlock(
                channels=Config.NUM_CHANNELS,
                dilation=dilation,
                dropout=Config.DROPOUT,
            )
            self.stage2_blocks.append(block)

        # --- Global Aggregation ---
        # BiGRU to capture global constraints
        # Input: 64. Hidden: 32 (Bidirectional -> 64 output)
        self.gru = nn.GRU(
            input_size=Config.NUM_CHANNELS,
            hidden_size=Config.NUM_CHANNELS // 2,
            batch_first=True,
            bidirectional=True,
        )

        # --- Output Head ---
        self.head = nn.Linear(Config.NUM_CHANNELS, 5)

    def forward(self, x, pair_indices):
        """
        Args:
            x: (B, L, 14) Input features
            pair_indices: (B, L) Indices of paired bases
        """
        x = x.permute(0, 2, 1)
        B, C, L = x.shape

        # Input Projection
        out = self.input_proj(x)

        # --- Stage 1 Forward ---
        for block in self.stage1_blocks:
            out = block(out)

        # Compression: (B, 32, L)
        compressed = self.compress(out)

        # --- Dynamic Gather (Pair-State) ---
        # 1. Pad compressed features with a zero vector at the end (index L)
        padding = torch.zeros(
            B, Config.BOTTLENECK_DIM, 1, device=x.device, dtype=x.dtype
        )
        compressed_padded = torch.cat([compressed, padding], dim=2)

        # 2. Adjust pair_indices: Replace -1 with L
        gather_idx = pair_indices.clone()
        gather_idx[gather_idx == -1] = L

        # 3. Expand indices
        gather_idx_expanded = gather_idx.unsqueeze(1).expand(
            -1, Config.BOTTLENECK_DIM, -1
        )

        # 4. Gather paired features
        paired_features = torch.gather(compressed_padded, 2, gather_idx_expanded)

        # 5. Concatenate Self + Paired features -> (B, 64, L)
        combined = torch.cat([compressed, paired_features], dim=1)

        # --- Stage 2 Forward ---
        out2 = combined
        for block in self.stage2_blocks:
            out2 = block(out2)

        # --- Global Aggregation ---
        out2 = out2.permute(0, 2, 1)
        gru_out, _ = self.gru(out2)

        # --- Head ---
        logits = self.head(gru_out)

        return logits
