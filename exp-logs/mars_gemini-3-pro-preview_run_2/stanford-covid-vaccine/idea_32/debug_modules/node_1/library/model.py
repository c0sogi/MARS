import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DilatedDenseBlock(nn.Module):
    """
    A single dilated convolution block with pre-activation and dropout.
    Designed for use in a DenseNet-style architecture where input channels grow
    due to the concatenation of previous feature maps.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super(DilatedDenseBlock, self).__init__()
        # Pre-activation structure: ReLU -> Conv -> ReLU -> Dropout
        # (Note: First ReLU is applied in forward)
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=dilation * (kernel_size - 1) // 2,
            dilation=dilation,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (Batch, In_Channels, Len)
        out = F.relu(x)
        out = self.conv(out)
        out = F.relu(out)
        out = self.dropout(out)
        return out


class NRDCN(nn.Module):
    """
    Normalized Recurrent Dense-Context Network (NR-DCN).

    Architecture:
    1. Input: Static Features + Tanh-Normalized Recycling Channels.
    2. Backbone: Dense Dilated TCN (DenseNet-style connections).
    3. Structural Interaction: Projection -> Partner Gather -> Null-Masking -> Concatenation.
    4. Aggregation: Bidirectional GRU.
    5. Output: Linear Projection.
    """

    def __init__(self):
        super(NRDCN, self).__init__()

        # ----------------------------------------------------------------------
        # Hyperparameters from Config
        # ----------------------------------------------------------------------
        self.input_channels = Config.INPUT_CHANNELS  # 23 (18 static + 5 dynamic)
        self.growth_rate = Config.GROWTH_RATE  # 64
        self.dilations = Config.DILATIONS  # [1, 2, 4, 8, 16, 32]
        self.dropout_rate = Config.DROPOUT  # 0.1
        self.kernel_size = Config.KERNEL_SIZE  # 3
        self.latent_dim = Config.LATENT_DIM  # 64
        self.gru_hidden = Config.GRU_HIDDEN_DIM  # 64
        self.num_targets = len(Config.TARGET_COLS)  # 5

        # ----------------------------------------------------------------------
        # Layers
        # ----------------------------------------------------------------------

        # 1. Initial Embedding
        # Maps the 23-channel input to the growth dimension
        self.embedding = nn.Conv1d(self.input_channels, self.growth_rate, kernel_size=1)

        # 2. Dense Dilated Backbone
        self.blocks = nn.ModuleList()
        current_channels = self.growth_rate

        for d in self.dilations:
            block = DilatedDenseBlock(
                in_channels=current_channels,
                out_channels=self.growth_rate,
                kernel_size=self.kernel_size,
                dilation=d,
                dropout=self.dropout_rate,
            )
            self.blocks.append(block)
            # In DenseNet, input to next layer is concatenation of all previous outputs
            current_channels += self.growth_rate

        # 3. Structural Interaction Projection
        # Projects the accumulated dense history to a compact latent dimension
        self.projection = nn.Conv1d(current_channels, self.latent_dim, kernel_size=1)

        # 4. Global Aggregation (BiGRU)
        # Input size is Latent (local) + Latent (partner)
        self.bigru = nn.GRU(
            input_size=self.latent_dim * 2,
            hidden_size=self.gru_hidden,
            batch_first=True,
            bidirectional=True,
        )

        # 5. Output Head
        # Input is GRU hidden * 2 (bidirectional)
        self.classifier = nn.Linear(self.gru_hidden * 2, self.num_targets)

    def forward(self, inputs, partner_indices, recycling=None):
        """
        Args:
            inputs (torch.Tensor): Static features (Batch, Seq_Len, 18).
            partner_indices (torch.Tensor): Indices of paired bases (Batch, Seq_Len).
            recycling (torch.Tensor, optional): Previous predictions (Batch, Seq_Len, 5).

        Returns:
            torch.Tensor: Predictions (Batch, Seq_Len, 5).
        """
        B, L, _ = inputs.shape

        # ----------------------------------------------------------------------
        # 1. Input Preparation & Recycling
        # ----------------------------------------------------------------------
        if recycling is None:
            # Cold Start: Initialize recycling channels with zeros
            recycling = torch.zeros(
                (B, L, self.num_targets), device=inputs.device, dtype=inputs.dtype
            )

        # Scale Alignment: Squash raw regression outputs to [-1, 1]
        # This prevents magnitude mismatch between one-hot inputs and regression feedback
        recycling = torch.tanh(recycling)

        # Concatenate static features (18) and normalized recycling (5) -> (23)
        x = torch.cat([inputs, recycling], dim=-1)

        # Permute to (Batch, Channels, Seq_Len) for Conv1d
        x = x.permute(0, 2, 1)

        # ----------------------------------------------------------------------
        # 2. Dense Dilated Backbone
        # ----------------------------------------------------------------------
        # Initialize feature list with the embedding output
        features = [self.embedding(x)]

        for block in self.blocks:
            # Dense Connection: Concatenate all previous features along channel dim
            inp = torch.cat(features, dim=1)
            out = block(inp)
            features.append(out)

        # ----------------------------------------------------------------------
        # 3. Structural Interaction
        # ----------------------------------------------------------------------
        # Concatenate full history from all blocks
        dense_out = torch.cat(features, dim=1)  # (B, Total_Channels, L)

        # Project to latent space
        latent = self.projection(dense_out)  # (B, 64, L)
        latent = latent.permute(0, 2, 1)  # (B, L, 64)

        # Gather Partner Features
        # partner_indices: (B, L). -1 indicates unpaired.
        mask = partner_indices != -1  # (B, L)

        # Replace -1 with 0 to ensure valid gather indices (we will mask the result later)
        safe_indices = partner_indices.masked_fill(~mask, 0)

        # Expand indices for gathering: (B, L, 1) -> (B, L, 64)
        gather_idx = safe_indices.unsqueeze(-1).expand(-1, -1, self.latent_dim)

        # Gather: For each position i, get the latent vector of its partner j
        partner_features = torch.gather(latent, 1, gather_idx)

        # Apply Null-Masking: Zero out features for unpaired positions
        partner_features = partner_features * mask.unsqueeze(-1).float()

        # Fusion: Concatenate local and partner features
        interaction_out = torch.cat([latent, partner_features], dim=-1)  # (B, L, 128)

        # ----------------------------------------------------------------------
        # 4. Global Aggregation & Output
        # ----------------------------------------------------------------------
        gru_out, _ = self.bigru(interaction_out)  # (B, L, 128)

        logits = self.classifier(gru_out)  # (B, L, 5)

        return logits
