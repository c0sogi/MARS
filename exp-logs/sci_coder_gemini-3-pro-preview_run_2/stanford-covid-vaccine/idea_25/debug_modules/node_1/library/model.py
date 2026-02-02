import torch
import torch.nn as nn
from library.config import Config


class DenseDilatedBlock(nn.Module):
    """
    A single dilated convolutional block for the DenseNet backbone.
    """

    def __init__(self, in_channels, growth_rate, kernel_size, dilation, dropout):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            growth_rate,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=(kernel_size - 1) * dilation // 2,
        )
        self.relu = nn.ReLU()
        self.norm = nn.BatchNorm1d(growth_rate)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        out = self.norm(out)
        out = self.dropout(out)
        return out


class LatentGather(nn.Module):
    """
    Projects features to a latent space and gathers partner features.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, partner_indices):
        # x: (B, C_in, L)
        # partner_indices: (B, L)

        # Project to latent dimension
        z = self.proj(x)  # (B, C_out, L)
        B, C, L = z.shape

        # Create mask for valid partners (partner_indices != -1)
        # partner_indices is -1 for unpaired bases
        mask = (partner_indices != -1).unsqueeze(1).float()  # (B, 1, L)

        # Replace -1 with 0 for gather operation (will be masked out later)
        safe_indices = partner_indices.clone()
        safe_indices[safe_indices == -1] = 0

        # Expand indices to match channel dimension: (B, C, L)
        idx_expanded = safe_indices.unsqueeze(1).expand(-1, C, -1)

        # Gather partner features
        z_partner = torch.gather(z, 2, idx_expanded)

        # Apply mask to zero out features for unpaired bases
        z_partner = z_partner * mask

        return z, z_partner


class RefinementModule(nn.Module):
    """
    Post-Interaction Refinement Module (Mini-DenseNet).
    Models stacking interactions on the fused pair features.
    """

    def __init__(self, in_channels, growth_rate, layers_config, dropout):
        super().__init__()
        self.layers = nn.ModuleList()
        current_channels = in_channels

        for kernel_size, dilation in layers_config:
            layer = nn.Sequential(
                nn.Conv1d(
                    current_channels,
                    growth_rate,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=(kernel_size - 1) * dilation // 2,
                ),
                nn.ReLU(),
                nn.BatchNorm1d(growth_rate),
                nn.Dropout(dropout),
            )
            self.layers.append(layer)
            # Input to next layer will include this layer's output (Dense connection)
            current_channels += growth_rate

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            # Concatenate all previous features
            inp = torch.cat(features, dim=1)
            out = layer(inp)
            features.append(out)

        # Return concatenation of all features (Input + L1 + L2 + ...)
        return torch.cat(features, dim=1)


class StackedInteractionDenseNet(nn.Module):
    """
    Main Architecture: Stacked Interaction Dense Network.
    Combines Dense Dilated TCN, Latent Interaction Gathering,
    Refinement Module, and BiGRU.
    """

    def __init__(self):
        super().__init__()

        # 1. Input Embedding
        # Maps raw input (19 channels) to the backbone growth rate
        self.embedding = nn.Sequential(
            nn.Conv1d(Config.INPUT_DIM, Config.BACKBONE_GROWTH_RATE, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(Config.BACKBONE_GROWTH_RATE),
        )

        # 2. Backbone: Dense Dilated TCN
        self.backbone_blocks = nn.ModuleList()
        current_channels = Config.BACKBONE_GROWTH_RATE

        for dilation in Config.BACKBONE_DILATIONS:
            block = DenseDilatedBlock(
                in_channels=current_channels,
                growth_rate=Config.BACKBONE_GROWTH_RATE,
                kernel_size=Config.BACKBONE_KERNEL_SIZE,
                dilation=dilation,
                dropout=Config.DROPOUT,
            )
            self.backbone_blocks.append(block)
            # Dense connection: next block takes all previous channels
            current_channels += Config.BACKBONE_GROWTH_RATE

        self.backbone_out_channels = current_channels

        # 3. Latent Interaction Gather
        self.latent_gather = LatentGather(
            in_channels=self.backbone_out_channels, out_channels=Config.LATENT_DIM
        )

        # 4. Refinement Module
        # Input is concatenation of Local (32) and Partner (32) -> 64
        refinement_in_channels = Config.LATENT_DIM * 2
        self.refinement = RefinementModule(
            in_channels=refinement_in_channels,
            growth_rate=Config.REFINEMENT_GROWTH_RATE,
            layers_config=Config.REFINEMENT_LAYERS,
            dropout=Config.DROPOUT,
        )

        # Calculate output dim of refinement: Input + (Num_Layers * Growth)
        refinement_out_channels = refinement_in_channels + (
            len(Config.REFINEMENT_LAYERS) * Config.REFINEMENT_GROWTH_RATE
        )

        # 5. Global Aggregation (BiGRU)
        self.rnn = nn.GRU(
            input_size=refinement_out_channels,
            hidden_size=Config.RNN_HIDDEN_DIM,
            num_layers=Config.RNN_LAYERS,
            batch_first=True,
            bidirectional=True,
        )

        # 6. Output Head
        # Bidirectional GRU outputs 2 * hidden_dim
        self.head = nn.Linear(Config.RNN_HIDDEN_DIM * 2, Config.NUM_TARGETS)

    def forward(self, inputs, partner_indices):
        """
        Args:
            inputs: (Batch, Seq_Len, Input_Dim)
            partner_indices: (Batch, Seq_Len)
        Returns:
            logits: (Batch, Seq_Len, Num_Targets)
        """
        # Permute for Conv1d: (B, L, C) -> (B, C, L)
        x = inputs.permute(0, 2, 1)

        # Embedding
        x = self.embedding(x)

        # Backbone (Dense Connections)
        features = [x]
        for block in self.backbone_blocks:
            # Input to block is concatenation of all prior features
            inp = torch.cat(features, dim=1)
            out = block(inp)
            features.append(out)

        # Concatenate all backbone features
        backbone_out = torch.cat(features, dim=1)

        # Latent Interaction
        z_local, z_partner = self.latent_gather(backbone_out, partner_indices)

        # Fusion: Concatenate Local + Partner
        fused = torch.cat([z_local, z_partner], dim=1)

        # Post-Interaction Refinement
        refined = self.refinement(fused)

        # Prepare for RNN: (B, C, L) -> (B, L, C)
        rnn_in = refined.permute(0, 2, 1)

        # RNN Aggregation
        rnn_out, _ = self.rnn(rnn_in)

        # Output Head
        logits = self.head(rnn_out)

        return logits
