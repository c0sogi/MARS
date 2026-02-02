import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DilatedDenseBlock(nn.Module):
    """
    A single block for the Dense Dilated Backbone.
    Performs Dilated Conv1d -> ReLU -> Dropout.
    Output is concatenated with input (Dense Connection).
    """

    def __init__(self, in_channels, growth_rate, dilation, dropout):
        super(DilatedDenseBlock, self).__init__()
        self.conv = nn.Conv1d(
            in_channels,
            growth_rate,
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
        return torch.cat([x, out], dim=1)


class StackingDenseRefinedNet(nn.Module):
    """
    Stacking-Aware Dense-Refined Hybrid Network.

    Architecture:
    1. Stage 1: Dense Dilated Backbone (Local Motif Extraction)
    2. Inter-Stage: Compression & Dynamic Pair-State Gathering
    3. Stage 2: Dense Dilated Backbone (Stacking Interaction Solver)
    4. Global Aggregation: Bidirectional GRU
    5. Head: Linear Projection
    """

    def __init__(self):
        super(StackingDenseRefinedNet, self).__init__()

        # --- Stage 1: Local Motif Extraction ---
        self.stage1_blocks = nn.ModuleList()
        current_dim = Config.INPUT_CHANNELS

        for dilation in Config.STAGE1_DILATIONS:
            block = DilatedDenseBlock(
                in_channels=current_dim,
                growth_rate=Config.GROWTH_RATE,
                dilation=dilation,
                dropout=Config.DROPOUT,
            )
            self.stage1_blocks.append(block)
            current_dim += Config.GROWTH_RATE

        self.stage1_out_dim = current_dim

        # --- Inter-Stage: Compression ---
        # Compresses high-dimensional dense features to a compact latent state
        self.compress = nn.Conv1d(self.stage1_out_dim, Config.HIDDEN_DIM, kernel_size=1)

        # --- Stage 2: Stacking Interaction Solver ---
        # Input dimension is HIDDEN_DIM * 2 (Self Features + Paired Features)
        self.stage2_blocks = nn.ModuleList()
        current_dim = Config.HIDDEN_DIM * 2

        for dilation in Config.STAGE2_DILATIONS:
            block = DilatedDenseBlock(
                in_channels=current_dim,
                growth_rate=Config.GROWTH_RATE,
                dilation=dilation,
                dropout=Config.DROPOUT,
            )
            self.stage2_blocks.append(block)
            current_dim += Config.GROWTH_RATE

        self.stage2_out_dim = current_dim

        # --- Global Aggregation ---
        # BiGRU to capture global constraints
        # Output dim is hidden_size * 2 (bidirectional), set to match input dim
        self.gru = nn.GRU(
            input_size=self.stage2_out_dim,
            hidden_size=self.stage2_out_dim // 2,
            batch_first=True,
            bidirectional=True,
        )

        # --- Output Head ---
        self.head = nn.Linear(self.stage2_out_dim, 5)

    def forward(self, x, pair_indices):
        """
        Args:
            x: (B, L, 14) Input features (Sequence, Structure, Loop)
            pair_indices: (B, L) Indices of paired bases (-1 for unpaired)

        Returns:
            logits: (B, L, 5) Predicted values
        """
        # Permute to (B, C, L) for Conv1d operations
        x = x.permute(0, 2, 1)
        B, C, L = x.shape

        # --- Stage 1 Forward ---
        out = x
        for block in self.stage1_blocks:
            out = block(out)

        # Compression: (B, HIDDEN_DIM, L)
        compressed = self.compress(out)

        # --- Dynamic Gather (Pair-State) ---
        # We want to concatenate feature[i] with feature[pair_indices[i]].
        # Unpaired bases have index -1. We handle this by padding the features
        # with a zero vector at index L, and mapping -1 to L.

        # 1. Pad compressed features with a zero vector at the end (index L)
        # Shape becomes (B, HIDDEN_DIM, L+1)
        padding = torch.zeros(B, Config.HIDDEN_DIM, 1, device=x.device, dtype=x.dtype)
        compressed_padded = torch.cat([compressed, padding], dim=2)

        # 2. Adjust pair_indices: Replace -1 with L
        # Shape (B, L)
        gather_idx = pair_indices.clone()
        gather_idx[gather_idx == -1] = L

        # 3. Expand indices to match channel dimension for torch.gather
        # Shape (B, HIDDEN_DIM, L)
        gather_idx_expanded = gather_idx.unsqueeze(1).expand(-1, Config.HIDDEN_DIM, -1)

        # 4. Gather paired features
        # Result: (B, HIDDEN_DIM, L)
        paired_features = torch.gather(compressed_padded, 2, gather_idx_expanded)

        # 5. Concatenate Self + Paired features
        # Result: (B, 2*HIDDEN_DIM, L)
        combined = torch.cat([compressed, paired_features], dim=1)

        # --- Stage 2 Forward ---
        out2 = combined
        for block in self.stage2_blocks:
            out2 = block(out2)

        # --- Global Aggregation ---
        # Permute back to (B, L, C) for RNN and Linear layers
        out2 = out2.permute(0, 2, 1)
        gru_out, _ = self.gru(out2)

        # --- Head ---
        logits = self.head(gru_out)

        return logits
