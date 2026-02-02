import torch
import torch.nn as nn
import torch.nn.functional as F
from library import config


class DenseDilatedBlock(nn.Module):
    """
    Single-Layer Dilated Residual Block with Dense Connections.
    """

    def __init__(self, in_channels, growth_rate, kernel_size, dilation, dropout):
        super(DenseDilatedBlock, self).__init__()
        self.conv = nn.Conv1d(
            in_channels, growth_rate, kernel_size, padding=dilation, dilation=dilation
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        out = self.dropout(out)
        return out


class StructuralInteractionBlock(nn.Module):
    """
    Latent Structural Interaction Block.
    Compresses features, gathers partner features, and fuses them.
    """

    def __init__(self, in_channels, compressed_channels=32):
        super(StructuralInteractionBlock, self).__init__()
        self.compress = nn.Conv1d(in_channels, compressed_channels, 1)

    def forward(self, x, partner_indices):
        """
        Args:
            x: [Batch, Channels, Length] - Backbone output
            partner_indices: [Batch, Length] - Indices of paired bases (-1 for unpaired)
        Returns:
            Fused features [Batch, Length, compressed_channels * 2]
        """
        # Compress: [B, C_in, L] -> [B, 32, L]
        compressed = self.compress(x)

        # Reshape for gathering: [B, L, 32]
        compressed = compressed.permute(0, 2, 1)
        B, L, C = compressed.shape

        # Flatten to [B*L, 32]
        flat_compressed = compressed.reshape(B * L, C)

        # Calculate global indices for gathering
        # partner_indices are local (0 to L-1). We need global (0 to B*L-1)
        batch_offsets = torch.arange(B, device=x.device).unsqueeze(1) * L
        flat_indices = partner_indices + batch_offsets
        flat_indices = flat_indices.view(-1)

        # Mask for unpaired bases
        mask = partner_indices.view(-1) != -1

        # Safe gather: replace -1 with 0 to avoid index error, then mask result
        safe_indices = flat_indices.clone()
        safe_indices[~mask] = 0

        gathered = flat_compressed[safe_indices]  # [B*L, 32]
        gathered[~mask] = 0.0  # Zero out unpaired

        # Reshape gathered back to [B, L, 32]
        gathered = gathered.view(B, L, C)

        # Fuse: Concatenate local compressed and gathered partner features
        # [B, L, 32] + [B, L, 32] -> [B, L, 64]
        out = torch.cat([compressed, gathered], dim=2)

        return out


class RecurrentDenseNet(nn.Module):
    """
    Recurrent Self-Correcting Dense Network.
    Uses Dense Dilated TCN backbone, Structural Interaction, and BiGRU.
    """

    def __init__(self):
        super(RecurrentDenseNet, self).__init__()

        # Configuration
        self.input_dim = config.INPUT_DIM
        self.growth_rate = config.GROWTH_RATE
        self.hidden_dim = config.HIDDEN_DIM
        self.dropout = config.DROPOUT
        self.kernel_size = config.KERNEL_SIZE

        # 1. Dense Backbone
        # Input: 23 channels (18 static + 5 recycled)
        self.blocks = nn.ModuleList()
        current_dim = self.input_dim
        dilations = [1, 2, 4, 8, 16, 32]

        for d in dilations:
            self.blocks.append(
                DenseDilatedBlock(
                    current_dim, self.growth_rate, self.kernel_size, d, self.dropout
                )
            )
            current_dim += self.growth_rate

        self.backbone_out_dim = current_dim

        # 2. Structural Interaction
        # Compresses backbone output and incorporates partner info
        self.interaction = StructuralInteractionBlock(self.backbone_out_dim, 32)

        # 3. Global Aggregation (BiGRU)
        # Input: 32 (local) + 32 (partner) = 64
        self.gru = nn.GRU(
            input_size=64,
            hidden_size=self.hidden_dim,
            batch_first=True,
            bidirectional=True,
        )

        # 4. Output Head
        # BiGRU output is hidden_dim * 2 (bidirectional)
        self.head = nn.Linear(self.hidden_dim * 2, 5)

    def forward(self, x_static, x_recycled, partner_indices):
        """
        Args:
            x_static: [Batch, Length, 18]
            x_recycled: [Batch, Length, 5]
            partner_indices: [Batch, Length]
        """
        # 1. Input Construction
        # Concatenate static features and recycled predictions
        x = torch.cat([x_static, x_recycled], dim=2)  # [B, L, 23]
        x = x.permute(0, 2, 1)  # [B, 23, L] for Conv1d

        # 2. Dense Backbone
        features = [x]
        for block in self.blocks:
            # Dense connection: concatenate all previous features
            concat_input = torch.cat(features, dim=1)
            out = block(concat_input)
            features.append(out)

        backbone_out = torch.cat(features, dim=1)  # [B, Total_Dim, L]

        # 3. Structural Interaction
        interaction_out = self.interaction(backbone_out, partner_indices)  # [B, L, 64]

        # 4. BiGRU
        gru_out, _ = self.gru(interaction_out)  # [B, L, 128]

        # 5. Output Head
        logits = self.head(gru_out)  # [B, L, 5]

        return logits
