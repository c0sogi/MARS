import torch
import torch.nn as nn
import math
from library.config import Config


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        # pe shape: (1, max_len, d_model)
        # Slicing pe to match seq_len
        return x + self.pe[:, : x.size(1), :]


class KinematicStream(nn.Module):
    """
    Processes the time-series window of kinematic features using a Transformer Encoder.
    Extracts the embedding corresponding to the center timestamp.
    """

    def __init__(self, input_dim, hidden_dim, num_layers, nhead, dropout):
        super(KinematicStream, self).__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(
            hidden_dim, max_len=Config.WINDOW_SIZE + 10
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.dropout = nn.Dropout(dropout)

        # Center index for extraction
        self.center_idx = Config.WINDOW_SIZE // 2

    def forward(self, x):
        # x shape: (batch_size, window_size, input_dim)

        # Project to hidden dimension
        x = self.input_proj(x)  # (B, L, H)

        # Add positional encoding
        x = self.pos_encoder(x)
        x = self.dropout(x)

        # Transformer Encoder
        # Output shape: (B, L, H)
        x = self.transformer_encoder(x)

        # Extract center token
        # We assume the window is constructed such that the target is at the center
        center_embedding = x[:, self.center_idx, :]  # (B, H)

        return center_embedding


class EnvironmentalStream(nn.Module):
    """
    Processes aggregated satellite geometry statistics (Sky-State) using an MLP.
    """

    def __init__(self, input_dim, hidden_dim, dropout):
        super(EnvironmentalStream, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x shape: (batch_size, input_dim)
        return self.mlp(x)


class SARTransformer(nn.Module):
    """
    Sky-Anchored Relative-State Transformer.
    Fuses Kinematic Stream (temporal context) and Environmental Stream (sky context)
    to predict metric residuals.
    """

    def __init__(
        self,
        kinematic_input_dim=len(Config.KINEMATIC_FEATURES),
        sky_input_dim=len(Config.SKY_FEATURES),
        output_dim=len(Config.TARGET_COLS),
        hidden_dim=Config.HIDDEN_DIM,
        num_layers=Config.NUM_LAYERS,
        nhead=Config.NHEAD,
        dropout=Config.DROPOUT,
    ):
        super(SARTransformer, self).__init__()

        self.kinematic_stream = KinematicStream(
            input_dim=kinematic_input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            nhead=nhead,
            dropout=dropout,
        )

        self.environmental_stream = EnvironmentalStream(
            input_dim=sky_input_dim, hidden_dim=hidden_dim, dropout=dropout
        )

        # Fusion Head
        # Concatenates the two streams (H + H = 2H) and projects to output
        self.fusion_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x_kin, x_sky):
        """
        Args:
            x_kin: Tensor of shape (batch_size, window_size, kinematic_features)
            x_sky: Tensor of shape (batch_size, sky_features)
        Returns:
            out: Tensor of shape (batch_size, 2) representing (dLat_m, dLon_m)
        """
        # Get embeddings
        emb_kin = self.kinematic_stream(x_kin)  # (B, H)
        emb_sky = self.environmental_stream(x_sky)  # (B, H)

        # Fuse
        fused = torch.cat([emb_kin, emb_sky], dim=1)  # (B, 2H)

        # Predict residuals
        out = self.fusion_head(fused)  # (B, 2)

        return out
