import torch
import torch.nn as nn
from library.config import Config


class DenseDilatedBlock(nn.Module):
    """
    A single dilated convolutional block that concatenates its output to its input
    (Dense Connection), allowing the network to grow in channel depth.
    """

    def __init__(self, in_channels, growth_rate, kernel_size, dilation, dropout):
        super(DenseDilatedBlock, self).__init__()
        # Calculate padding to maintain sequence length
        # For k=3, p = (2 * d) / 2 = d
        padding = (kernel_size - 1) * dilation // 2

        self.conv = nn.Conv1d(
            in_channels,
            growth_rate,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Apply ReLU -> Conv -> Dropout
        out = self.relu(x)
        out = self.conv(out)
        out = self.dropout(out)

        # Dense connection: Concatenate input and output along channel dimension
        return torch.cat([x, out], dim=1)


class InteractionEnrichmentModule(nn.Module):
    """
    Compresses features, gathers partner features, and computes explicit interactions
    (Product and Difference) to enrich the representation with physical context.
    """

    def __init__(self, in_channels, latent_dim):
        super(InteractionEnrichmentModule, self).__init__()
        self.project = nn.Conv1d(in_channels, latent_dim, kernel_size=1)

    def forward(self, x, partner_indices):
        """
        Args:
            x: (Batch, Channels, SeqLen)
            partner_indices: (Batch, SeqLen) - LongTensor containing indices of paired bases
        """
        # 1. Compress to latent dim
        z = self.project(x)  # (B, latent, L)

        # 2. Gather Partner Features
        B, C, L = z.shape
        # Expand indices to match channel dimension: (B, L) -> (B, 1, L) -> (B, C, L)
        idx_expanded = partner_indices.unsqueeze(1).expand(-1, C, -1)

        # Gather: out[b, c, i] = z[b, c, idx_expanded[b, c, i]]
        z_partner = torch.gather(z, 2, idx_expanded)

        # 3. Compute Interactions
        # Element-wise product (Similarity/Compatibility)
        z_mult = z * z_partner
        # Absolute difference (Asymmetry/Directionality)
        z_diff = torch.abs(z - z_partner)

        # 4. Concatenate all components
        # Output dim = 4 * latent_dim
        out = torch.cat([z, z_partner, z_mult, z_diff], dim=1)

        return out


class InteractionEnrichedDenseNet(nn.Module):
    def __init__(self):
        super(InteractionEnrichedDenseNet, self).__init__()

        # --- Configuration ---
        input_dim = Config.INPUT_DIM
        growth_rate = Config.GROWTH_RATE
        kernel_size = Config.KERNEL_SIZE
        dilations = Config.DILATIONS
        dropout = Config.DROPOUT
        latent_dim = Config.LATENT_DIM
        gru_hidden = Config.GRU_HIDDEN_DIM
        gru_layers = Config.GRU_LAYERS

        # --- 1. Initial Embedding ---
        # Projects input features to the growth rate dimension to start the dense block
        self.embedding = nn.Conv1d(input_dim, growth_rate, kernel_size=1)

        # --- 2. Dense Dilated Backbone ---
        self.blocks = nn.ModuleList()
        current_dim = growth_rate

        for d in dilations:
            block = DenseDilatedBlock(
                in_channels=current_dim,
                growth_rate=growth_rate,
                kernel_size=kernel_size,
                dilation=d,
                dropout=dropout,
            )
            self.blocks.append(block)
            current_dim += growth_rate

        # --- 3. Interaction Enrichment ---
        self.interaction = InteractionEnrichmentModule(current_dim, latent_dim)
        # Output of interaction module is 4 * latent_dim
        interaction_out_dim = 4 * latent_dim

        # --- 4. Global Aggregation (BiGRU) ---
        self.gru = nn.GRU(
            input_size=interaction_out_dim,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
        )

        # --- 5. Output Head ---
        # Bidirectional GRU outputs 2 * hidden_size
        self.head = nn.Linear(gru_hidden * 2, 5)

    def forward(self, x, partner_indices):
        """
        Args:
            x: (Batch, SeqLen, InputDim)
            partner_indices: (Batch, SeqLen)
        Returns:
            logits: (Batch, SeqLen, 5)
        """
        # Permute for Conv1d: (B, L, C) -> (B, C, L)
        x = x.transpose(1, 2)

        # Backbone
        out = self.embedding(x)
        for block in self.blocks:
            out = block(out)

        # Interaction Enrichment
        # out is (B, C_accumulated, L)
        out = self.interaction(out, partner_indices)
        # out is (B, 128, L)

        # GRU Processing
        # Permute back for GRU: (B, C, L) -> (B, L, C)
        out = out.transpose(1, 2)

        self.gru.flatten_parameters()
        out, _ = self.gru(out)

        # Prediction Head
        logits = self.head(out)

        return logits
