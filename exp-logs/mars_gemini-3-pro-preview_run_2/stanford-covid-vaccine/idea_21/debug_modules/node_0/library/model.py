import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DilatedResidualBlock(nn.Module):
    """
    A block consisting of a dilated 1D convolution, BatchNorm, ReLU, and Dropout.
    Used as the building block for the Dense TCN backbone.
    """

    def __init__(self, in_channels, out_channels, dilation):
        super(DilatedResidualBlock, self).__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(Config.DROPOUT)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)
        return out


class ScaleDecoupledDenseNet(nn.Module):
    """
    The main architecture implementing the Scale-Decoupled Compact Dense Network.
    It features a Dense Dilated TCN backbone, dual-stream (local/global) latent compression,
    partner feature gathering, and BiGRU aggregation.
    """

    def __init__(self):
        super(ScaleDecoupledDenseNet, self).__init__()

        # Backbone: Dense Dilated TCN
        self.blocks = nn.ModuleList()
        current_dim = Config.INPUT_CHANNELS

        # Dilations: 1, 2, 4 (Local) | 8, 16, 32 (Global)
        for d in Config.DILATIONS:
            # Growth Rate is fixed to Config.HIDDEN_DIM (64)
            block = DilatedResidualBlock(current_dim, Config.HIDDEN_DIM, d)
            self.blocks.append(block)
            # Dense Connection: Input to next layer grows by block output size
            current_dim += Config.HIDDEN_DIM

        # Compression Layers (1x1 Convs)
        # Local stream: 3 blocks * 64 = 192 channels (from dilations 1, 2, 4)
        self.local_compress = nn.Conv1d(
            3 * Config.HIDDEN_DIM, Config.LATENT_DIM, kernel_size=1
        )

        # Global stream: 3 blocks * 64 = 192 channels (from dilations 8, 16, 32)
        self.global_compress = nn.Conv1d(
            3 * Config.HIDDEN_DIM, Config.LATENT_DIM, kernel_size=1
        )

        # Aggregation (BiGRU)
        # Input: Local(32) + Global(32) + PartnerLocal(32) + PartnerGlobal(32) = 128
        gru_input_dim = 4 * Config.LATENT_DIM
        self.gru = nn.GRU(
            input_size=gru_input_dim,
            hidden_size=Config.HIDDEN_DIM,  # 64
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # Output Head
        # BiGRU output: 64 * 2 = 128
        self.head = nn.Linear(Config.HIDDEN_DIM * 2, 5)

    def forward(self, x, partner_indices):
        """
        Args:
            x (torch.Tensor): Input features of shape (Batch, Channels, Length).
            partner_indices (torch.Tensor): Indices of paired bases, shape (Batch, Length).
                                            Unpaired bases are marked with -1.
        Returns:
            torch.Tensor: Predictions of shape (Batch, Length, 5).
        """
        block_outputs = []
        current_input = x

        # 1. Backbone Pass with Dense Connections
        for block in self.blocks:
            out = block(current_input)
            block_outputs.append(out)
            # Concatenate input and output for the next layer
            current_input = torch.cat([current_input, out], dim=1)

        # 2. Scale Decoupling
        # Local: blocks 0, 1, 2 (dilations 1, 2, 4)
        local_feats = torch.cat(block_outputs[:3], dim=1)  # (B, 192, L)
        z_local = self.local_compress(local_feats)  # (B, 32, L)

        # Global: blocks 3, 4, 5 (dilations 8, 16, 32)
        global_feats = torch.cat(block_outputs[3:], dim=1)  # (B, 192, L)
        z_global = self.global_compress(global_feats)  # (B, 32, L)

        # 3. Interaction (Gather Partner Features)
        # Permute to (B, L, C) for gathering
        z_local = z_local.permute(0, 2, 1)  # (B, L, 32)
        z_global = z_global.permute(0, 2, 1)  # (B, L, 32)

        B, L, C = z_local.shape

        # Handle unpaired indices (-1).
        # We append a zero vector at index L for each batch to serve as the "unpaired" embedding.
        dummy = torch.zeros(B, 1, C, device=x.device)

        z_local_padded = torch.cat([z_local, dummy], dim=1)  # (B, L+1, 32)
        z_global_padded = torch.cat([z_global, dummy], dim=1)  # (B, L+1, 32)

        # Map -1 indices to L (the index of the dummy vector)
        gather_indices = partner_indices.clone()
        gather_indices[gather_indices == -1] = L

        # Expand indices for gathering across the channel dimension: (B, L, C)
        gather_indices_expanded = gather_indices.unsqueeze(-1).expand(-1, -1, C)

        # Gather partner features
        p_local = torch.gather(z_local_padded, 1, gather_indices_expanded)
        p_global = torch.gather(z_global_padded, 1, gather_indices_expanded)

        # Fusion: Concatenate self and partner features
        # Shape: (B, L, 128)
        fused = torch.cat([z_local, z_global, p_local, p_global], dim=2)

        # 4. Aggregation
        gru_out, _ = self.gru(fused)  # (B, L, 128)

        # 5. Head
        logits = self.head(gru_out)  # (B, L, 5)

        return logits
