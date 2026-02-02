import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import MODEL_PARAMS


class DilatedDenseBlock(nn.Module):
    """
    A single-layer dilated convolutional block designed for a DenseNet-style backbone.

    Args:
        in_channels (int): Number of input channels (accumulated from previous blocks).
        growth_rate (int): Number of output channels for this block.
        kernel_size (int): Size of the convolving kernel.
        dilation (int): Spacing between kernel elements.
        dropout (float): Dropout probability.
    """

    def __init__(self, in_channels, growth_rate, kernel_size, dilation, dropout):
        super(DilatedDenseBlock, self).__init__()
        # Padding to maintain sequence length: padding = dilation * (kernel_size - 1) / 2
        padding = dilation * (kernel_size - 1) // 2

        self.conv = nn.Conv1d(
            in_channels, growth_rate, kernel_size, padding=padding, dilation=dilation
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Standard Conv -> ReLU -> Dropout
        out = F.relu(self.conv(x))
        out = self.dropout(out)
        return out


class LatentInteraction(nn.Module):
    """
    Modeling structural interactions by gathering features from paired bases.

    Steps:
    1. Project input features to latent dimension.
    2. Gather features from partner indices.
    3. Mask gathered features where the base is unpaired.
    4. Concatenate local and gathered features.
    """

    def __init__(self, in_channels, latent_dim):
        super(LatentInteraction, self).__init__()
        self.project = nn.Conv1d(in_channels, latent_dim, 1)

    def forward(self, x, partner_indices, unpaired_mask):
        """
        Args:
            x (torch.Tensor): Input features (B, C, L).
            partner_indices (torch.Tensor): Indices of paired bases (B, L).
            unpaired_mask (torch.Tensor): Boolean mask (B, L) where True indicates unpaired.
        """
        # 1. Project
        z = self.project(x)  # (B, latent_dim, L)
        B, C, L = z.shape

        # 2. Gather Partner Features
        # Permute to (B, L, C) for easier indexing
        z_perm = z.permute(0, 2, 1)
        flat_z = z_perm.reshape(B * L, C)

        # Calculate flat indices: batch_offset + partner_index
        batch_idx = torch.arange(B, device=z.device).unsqueeze(1).expand(B, L)
        flat_indices = (batch_idx * L + partner_indices).view(-1)

        # Gather and reshape back to (B, L, C) -> (B, C, L)
        z_partner_flat = flat_z[flat_indices]
        z_partner = z_partner_flat.view(B, L, C).permute(0, 2, 1)

        # 3. Null-Masking
        # If a base is unpaired, its partner index is invalid (or points to 0).
        # We explicitly zero out the gathered vector for unpaired bases.
        if unpaired_mask is not None:
            # Expand mask to (B, C, L)
            mask_expanded = unpaired_mask.unsqueeze(1).expand_as(z_partner)
            z_partner = z_partner.masked_fill(mask_expanded, 0.0)

        # 4. Fusion (Concatenation)
        # Output dim: latent_dim * 2
        out = torch.cat([z, z_partner], dim=1)
        return out


class SR_DCN(nn.Module):
    """
    Stabilized Recurrent Dense-Context Network.

    Combines a Dense Dilated TCN backbone with a Structural Interaction layer
    and a BiGRU for global context. Designed to work with recycling channels.
    """

    def __init__(self):
        super(SR_DCN, self).__init__()
        config = MODEL_PARAMS

        self.input_dim = config["input_dim"]
        self.hidden_dim = config["hidden_dim"]
        self.dilation_rates = config["dilation_rates"]
        self.dropout_rate = config["dropout"]
        self.num_targets = config["num_targets"]
        self.kernel_size = config["kernel_size"]

        # --- Backbone: Dense Dilated TCN ---
        self.blocks = nn.ModuleList()
        current_dim = self.input_dim

        for d in self.dilation_rates:
            block = DilatedDenseBlock(
                in_channels=current_dim,
                growth_rate=self.hidden_dim,
                kernel_size=self.kernel_size,
                dilation=d,
                dropout=self.dropout_rate,
            )
            self.blocks.append(block)
            # In DenseNet, input to next layer is concatenation of all previous
            current_dim += self.hidden_dim

        self.backbone_out_dim = current_dim

        # --- Latent Structural Interaction ---
        # Projects dense history to hidden_dim (64), then fuses with partner (64) -> 128
        self.interaction = LatentInteraction(self.backbone_out_dim, self.hidden_dim)
        self.interaction_out_dim = self.hidden_dim * 2

        # --- Global Aggregation: BiGRU ---
        # Input: 128, Hidden: 64. Bidirectional output: 128.
        self.gru = nn.GRU(
            input_size=self.interaction_out_dim,
            hidden_size=self.hidden_dim,
            batch_first=True,
            bidirectional=True,
        )

        # --- Output Head ---
        self.head = nn.Linear(self.hidden_dim * 2, self.num_targets)

    def forward(self, x, partner_indices):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, L, input_dim).
                              Includes static features + recycling channels.
            partner_indices (torch.Tensor): Indices of paired bases (B, L).

        Returns:
            torch.Tensor: Predictions of shape (B, L, num_targets).
        """
        # Permute to (B, C, L) for Conv1d
        x_in = x.permute(0, 2, 1)

        # 1. Backbone (Dense Connections)
        features = [x_in]
        for block in self.blocks:
            # Concatenate all previous features along channel dimension
            dense_input = torch.cat(features, dim=1)
            out = block(dense_input)
            features.append(out)

        # Final dense representation
        backbone_out = torch.cat(features, dim=1)  # (B, backbone_out_dim, L)

        # 2. Extract Unpaired Mask
        # Structure vocab is "().". '.' is at index 2.
        # Input channels: Seq(4) + Struct(3) ...
        # Struct channels start at index 4. So '.' is at index 4+2 = 6.
        # Check if the value at channel 6 is active (1.0)
        unpaired_mask = x_in[:, 6, :] > 0.5  # (B, L)

        # 3. Latent Interaction
        interaction_out = self.interaction(
            backbone_out, partner_indices, unpaired_mask
        )  # (B, 128, L)

        # 4. Global Aggregation (BiGRU)
        # Permute back to (B, L, C) for RNN
        rnn_in = interaction_out.permute(0, 2, 1)
        rnn_out, _ = self.gru(rnn_in)  # (B, L, 128)

        # 5. Head
        logits = self.head(rnn_out)  # (B, L, 5)

        return logits
