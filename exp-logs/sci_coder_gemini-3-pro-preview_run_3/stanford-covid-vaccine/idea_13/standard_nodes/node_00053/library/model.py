import torch
import torch.nn as nn
from library.config import Config


class GatedSpatialInjection(nn.Module):
    """
    Implements the Gated Latent Spatial Injection mechanism.
    It uses the secondary structure pair indices to retrieve features from paired bases,
    computes a 'trust' gate based on the concatenation of local and paired features,
    and injects the paired information into the local representation.
    """

    def __init__(self, dim):
        super(GatedSpatialInjection, self).__init__()
        # Projection for the paired feature
        self.proj = nn.Linear(dim, dim)
        # Gating mechanism: takes [h_i; h_j] -> scalar gate per channel
        # "trust gate g_ij = sigmoid(W_gate . [h_i; h_j] + b_gate)"
        self.gate = nn.Linear(dim * 2, dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pair_indices):
        """
        Args:
            x (torch.Tensor): Input features of shape (Batch, Length, Channels).
            pair_indices (torch.Tensor): Indices of paired bases of shape (Batch, Length).
                                         -1 indicates unpaired.
        Returns:
            torch.Tensor: Spatially enriched features of shape (Batch, Length, Channels).
        """
        B, L, C = x.shape

        # 1. Create a mask for valid pairs (1 if paired, 0 if unpaired)
        # pair_indices is (B, L)
        mask = (pair_indices != -1).unsqueeze(-1).float()  # (B, L, 1)

        # 2. Handle invalid indices for gather
        # Replace -1 with 0 to prevent index out of bounds errors.
        # The result will be masked out anyway.
        safe_indices = pair_indices.clone()
        safe_indices[pair_indices == -1] = 0

        # 3. Gather paired features
        # We need to gather along the sequence dimension (dim=1).
        # Expand indices to match channel dimension: (B, L, C)
        gather_indices = safe_indices.unsqueeze(-1).expand(-1, -1, C)

        # Gather: out[b, i, c] = input[b, index[b, i, c], c]
        # Here index[b, i, c] is the paired position j. So we get x[b, j, c].
        x_paired = torch.gather(x, 1, gather_indices)

        # Mask out features where there was no pair
        x_paired = x_paired * mask

        # 4. Compute Gate
        # Concatenate local features h_i and paired features h_j
        concat_features = torch.cat([x, x_paired], dim=-1)  # (B, L, 2*C)
        gates = self.sigmoid(self.gate(concat_features))  # (B, L, C)

        # 5. Compute Update
        # Project paired features: W_proj * h_j
        update = self.proj(x_paired)  # (B, L, C)

        # 6. Apply Injection
        # h'_i = h_i + g_{ij} * (W_proj * h_j)
        # If unpaired, mask is 0 -> x_paired is 0 -> update is 0 -> no change.
        out = x + gates * update

        return out


class GatedSpatialConvBiGRU(nn.Module):
    """
    The main architecture combining a 1D Convolutional Stem,
    Gated Spatial Injection, and a high-capacity BiGRU backbone.
    """

    def __init__(self, config=Config):
        super(GatedSpatialConvBiGRU, self).__init__()
        self.config = config

        # ==============================
        # 1. Convolutional Stem
        # ==============================
        # Projects sparse one-hot inputs to dense local features
        self.conv = nn.Conv1d(
            in_channels=config.INPUT_CHANNELS,
            out_channels=config.CONV_FILTERS,
            kernel_size=config.CONV_KERNEL,
            padding=config.CONV_KERNEL // 2,  # Same padding
        )
        self.act = nn.GELU()
        self.dropout_stem = nn.Dropout(config.DROPOUT)

        # ==============================
        # 2. Gated Spatial Injection
        # ==============================
        # Injects structural dependencies dynamically
        self.spatial_injection = GatedSpatialInjection(config.CONV_FILTERS)

        # ==============================
        # 3. Recurrent Backbone
        # ==============================
        # BiGRU for long-range sequence modeling
        self.rnn = nn.GRU(
            input_size=config.CONV_FILTERS,
            hidden_size=config.HIDDEN_DIM,
            num_layers=config.NUM_LAYERS,
            dropout=config.DROPOUT if config.NUM_LAYERS > 1 else 0,
            bidirectional=config.BIDIRECTIONAL,
            batch_first=True,
        )

        # ==============================
        # 4. Output Head
        # ==============================
        rnn_out_dim = (
            config.HIDDEN_DIM * 2 if config.BIDIRECTIONAL else config.HIDDEN_DIM
        )
        self.head = nn.Linear(rnn_out_dim, config.NUM_TARGETS)

    def forward(self, inputs, pair_indices):
        """
        Args:
            inputs (torch.Tensor): (Batch, Length, Channels)
            pair_indices (torch.Tensor): (Batch, Length)

        Returns:
            torch.Tensor: Predictions (Batch, Length, Num_Targets)
        """
        # 1. Stem
        # Permute to (B, C, L) for Conv1d
        x = inputs.permute(0, 2, 1)
        x = self.conv(x)
        x = self.act(x)
        x = self.dropout_stem(x)

        # Permute back to (B, L, C)
        x = x.permute(0, 2, 1)

        # 2. Spatial Injection
        # Inject structural context into the local convolutional features
        x = self.spatial_injection(x, pair_indices)

        # 3. Backbone
        # GRU expects (B, L, H_in)
        x, _ = self.rnn(x)

        # 4. Head
        # Project to targets
        out = self.head(x)

        return out
