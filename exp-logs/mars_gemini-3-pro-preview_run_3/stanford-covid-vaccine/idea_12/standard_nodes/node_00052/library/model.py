import torch
import torch.nn as nn
from library.config import Config


class SpatialInjection(nn.Module):
    """
    Injects spatial context by concatenating features from paired bases.
    If a base is unpaired, the 'paired' feature is masked to zero.
    """

    def __init__(self, input_dim):
        super().__init__()
        # Projects concatenated [local, paired] vectors (input_dim * 2) back to input_dim
        self.proj = nn.Linear(input_dim * 2, input_dim)

    def forward(self, x, pair_indices):
        """
        Args:
            x: Tensor of shape (Batch, SeqLen, Dim)
            pair_indices: LongTensor of shape (Batch, SeqLen)
        """
        batch_size, seq_len, dim = x.shape
        device = x.device

        # 1. Gather paired features
        # Create batch indices for advanced indexing: (Batch, SeqLen)
        batch_idx = (
            torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, seq_len)
        )

        # Gather features: x[batch, pair_index, :]
        # This retrieves the feature vector of the base paired with the current position
        paired_features = x[batch_idx, pair_indices, :]

        # 2. Mask unpaired positions
        # In the data processing, unpaired bases map to themselves (pair_index == current_index).
        # The prompt requires unpaired positions to use a zero vector for the paired feature.
        seq_idx = (
            torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        )

        # Mask is 1.0 where paired (indices differ), 0.0 where unpaired (indices same)
        mask = (pair_indices != seq_idx).unsqueeze(-1).type_as(x)

        # Apply mask
        paired_features = paired_features * mask

        # 3. Concatenate and Project
        # Concatenate local features (x) and paired features
        combined = torch.cat([x, paired_features], dim=-1)

        # Project back to original dimension
        out = self.proj(combined)

        return out


class LatentSpatialBiGRU(nn.Module):
    """
    Latent Spatial-Contextualized BiGRU Architecture.

    Components:
    1. 1D Convolutional Stem (Local Feature Extraction)
    2. Latent Spatial Injection (Structural Context)
    3. BiGRU Backbone (Sequence Modeling)
    4. Linear Head (Prediction)
    """

    def __init__(self, config=None):
        super().__init__()
        if config is None:
            config = Config()

        # Hyperparameters
        self.input_channels = config.input_channels
        self.conv_filters = config.conv_filters
        self.kernel_size = config.kernel_size
        self.hidden_dim = config.hidden_dim
        self.n_layers = config.n_layers
        self.dropout_p = config.dropout
        self.output_dim = config.output_dim

        # 1. Convolutional Stem
        # Projects sparse one-hot inputs to dense latent space and aggregates local context.
        # Input: (Batch, Channels, SeqLen)
        self.conv_stem = nn.Sequential(
            nn.Conv1d(
                in_channels=self.input_channels,
                out_channels=self.conv_filters,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2,
            ),
            nn.GELU(),
            nn.Dropout(self.dropout_p),
        )

        # 2. Latent Spatial Injection
        # Mixes structural dependencies into the latent sequence representation.
        self.spatial_injection = SpatialInjection(self.conv_filters)
        self.spatial_dropout = nn.Dropout(self.dropout_p)

        # 3. Backbone (BiGRU)
        # High-capacity recurrent network to model long-range dependencies.
        self.gru = nn.GRU(
            input_size=self.conv_filters,
            hidden_size=self.hidden_dim,
            num_layers=self.n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=self.dropout_p if self.n_layers > 1 else 0.0,
        )

        # 4. Output Head
        # Projects from BiGRU hidden state (2 * hidden_dim) to targets.
        self.head = nn.Linear(self.hidden_dim * 2, self.output_dim)

    def forward(self, x, pair_indices):
        """
        Args:
            x: Input tensor (Batch, SeqLen, InputChannels)
            pair_indices: Pairing indices (Batch, SeqLen)

        Returns:
            out: Prediction tensor (Batch, SeqLen, OutputDim)
        """
        # Permute for Conv1d: (N, L, C) -> (N, C, L)
        x = x.permute(0, 2, 1)

        # Apply Convolutional Stem
        x = self.conv_stem(x)

        # Permute back for Spatial Injection and RNN: (N, C, L) -> (N, L, C)
        x = x.permute(0, 2, 1)

        # Apply Spatial Injection
        x = self.spatial_injection(x, pair_indices)
        x = self.spatial_dropout(x)

        # Apply BiGRU Backbone
        # gru_out shape: (Batch, SeqLen, 2 * HiddenDim)
        gru_out, _ = self.gru(x)

        # Apply Output Head
        out = self.head(gru_out)

        return out
