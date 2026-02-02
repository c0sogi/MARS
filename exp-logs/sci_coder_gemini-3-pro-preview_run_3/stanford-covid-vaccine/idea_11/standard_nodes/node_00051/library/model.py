import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class LatentSpatialMixer(nn.Module):
    """
    Augments local features with structural context in the latent space.
    Retrieves feature vectors from paired bases and fuses them with the current position's features.
    """

    def __init__(self, input_dim, output_dim, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim

        # Project concatenated features (local + paired) back to desired dimension
        # Input is 2 * input_dim because we concat x_i and x_j
        self.projection = nn.Sequential(
            nn.Linear(input_dim * 2, output_dim), nn.GELU(), nn.Dropout(dropout)
        )

    def forward(self, x, pair_index):
        """
        Args:
            x: Tensor of shape (Batch, Channels, Seq_Len) representing local features.
            pair_index: Tensor of shape (Batch, Seq_Len) with indices of paired bases (-1 if unpaired).

        Returns:
            Tensor of shape (Batch, Output_Dim, Seq_Len)
        """
        B, C, L = x.shape

        # 1. Prepare indices for gathering
        # Replace -1 with 0 to allow valid indexing (we will mask these out later)
        # Shape: (B, L)
        safe_indices = pair_index.clone()
        mask = (safe_indices != -1).float().unsqueeze(1)  # (B, 1, L)
        safe_indices[safe_indices == -1] = 0

        # Expand indices to match channel dimension: (B, C, L)
        gather_indices = safe_indices.unsqueeze(1).expand(-1, C, -1)

        # 2. Gather paired features
        # x is (B, C, L). We want to gather along dim 2 (Seq_Len).
        # paired_features[b, c, i] = x[b, c, pair_index[b, i]]
        paired_x = torch.gather(x, 2, gather_indices.long())

        # 3. Apply mask to zero out features gathered from dummy index 0 for unpaired bases
        paired_x = paired_x * mask

        # 4. Feature Fusion
        # Permute to (B, L, C) for Linear layer application
        x_perm = x.permute(0, 2, 1)  # (B, L, C)
        paired_perm = paired_x.permute(0, 2, 1)  # (B, L, C)

        # Concatenate: (B, L, 2*C)
        combined = torch.cat([x_perm, paired_perm], dim=-1)

        # Project: (B, L, Out_Dim)
        out = self.projection(combined)

        # Permute back to (B, Out_Dim, L) to maintain consistency with Conv layers if needed,
        # or keep as is. The backbone expects (B, L, C), so we return (B, L, Out_Dim).
        # However, to be consistent with the input shape convention of this block (B, C, L),
        # let's return (B, Out_Dim, L) and let the model handle permutation for RNN.
        return out.permute(0, 2, 1)


class RNAModel(nn.Module):
    """
    Latent Spatially-Augmented BiGRU Model.

    Architecture:
    1. 1D Convolutional Stem (Local Feature Extraction)
    2. Latent Spatial Mixer (Structural Augmentation)
    3. Bidirectional GRU Backbone (Sequence Modeling)
    4. Linear Head (Prediction)
    """

    def __init__(self, config=None):
        super().__init__()
        # Use provided config or default to library.config.Config
        if config is None:
            config = Config

        self.seq_len = config.seq_len
        self.pred_len = config.pred_len

        # --- 1. Convolutional Stem ---
        # Projects sparse one-hot inputs (14 channels) to dense latent space (256 channels)
        # Kernel size 3 aggregates immediate k-mer context.
        self.conv_stem = nn.Sequential(
            nn.Conv1d(
                in_channels=config.input_channels,
                out_channels=config.conv_filters,
                kernel_size=config.conv_kernel,
                padding=config.conv_kernel // 2,
            ),
            nn.GELU(),
            nn.BatchNorm1d(config.conv_filters),
            nn.Dropout(config.dropout),
        )

        # --- 2. Latent Spatial Mixer ---
        # Fuses local features with paired features.
        # We keep the dimension consistent (conv_filters -> conv_filters)
        self.spatial_mixer = LatentSpatialMixer(
            input_dim=config.conv_filters,
            output_dim=config.conv_filters,
            dropout=config.dropout,
        )

        # --- 3. Backbone (BiGRU) ---
        # High-capacity RNN.
        self.rnn = nn.GRU(
            input_size=config.conv_filters,
            hidden_size=config.rnn_hidden_dim,
            num_layers=config.rnn_layers,
            dropout=config.dropout if config.rnn_layers > 1 else 0,
            bidirectional=True,
            batch_first=True,
        )

        # --- 4. Output Head ---
        # Projects from RNN hidden state (2 * hidden_dim) to targets (5)
        self.head = nn.Sequential(
            nn.Linear(config.rnn_hidden_dim * 2, config.rnn_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.rnn_hidden_dim, config.num_targets),
        )

    def forward(self, inputs, pair_index, **kwargs):
        """
        Args:
            inputs: Tensor of shape (Batch, Seq_Len, Channels=14)
            pair_index: Tensor of shape (Batch, Seq_Len)

        Returns:
            Tensor of shape (Batch, Seq_Len, Num_Targets=5)
        """
        # Permute inputs to (Batch, Channels, Seq_Len) for Conv1d
        x = inputs.permute(0, 2, 1)

        # 1. Local Feature Extraction
        x = self.conv_stem(x)  # (B, Filters, L)

        # 2. Latent Spatial Mixing
        # Injects structural info. Output is (B, Filters, L)
        x = self.spatial_mixer(x, pair_index)

        # 3. Sequence Modeling
        # Permute to (Batch, Seq_Len, Filters) for RNN
        x = x.permute(0, 2, 1)

        # RNN returns (output, h_n). We only need output.
        # output shape: (Batch, Seq_Len, 2 * Hidden_Dim)
        x, _ = self.rnn(x)

        # 4. Prediction
        out = self.head(x)  # (Batch, Seq_Len, 5)

        return out
