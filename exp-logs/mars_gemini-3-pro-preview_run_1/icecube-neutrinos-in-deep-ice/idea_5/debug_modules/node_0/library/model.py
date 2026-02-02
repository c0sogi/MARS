import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from library.config import Config


class FourierFeatureEncoding(nn.Module):
    """
    Projects inputs into a higher dimensional space using Fourier features.
    Used for encoding spatiotemporal coordinates (x, y, z, t).
    """

    def __init__(self, input_dim=4, output_dim=128, num_frequencies=16, scale=1.0):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_frequencies = num_frequencies
        self.scale = scale

        # Define frequencies: 2^0, 2^1, ..., 2^(num_frequencies-1)
        self.register_buffer(
            "frequencies", 2.0 ** torch.arange(num_frequencies, dtype=torch.float32)
        )

        # Calculate the dimension after fourier expansion
        # For each input channel, we have sin and cos for each frequency
        # dim = input_dim * num_frequencies * 2
        fourier_dim = input_dim * num_frequencies * 2

        # Projection layer to map back to model_dim
        self.projection = nn.Linear(fourier_dim, output_dim)

    def forward(self, x):
        """
        Args:
            x: (Batch, Seq, input_dim)
        Returns:
            (Batch, Seq, output_dim)
        """
        # x shape: (B, N, C)
        # frequencies shape: (F,)

        # Expand x to (B, N, C, 1) and frequencies to (1, 1, 1, F)
        x_expanded = x.unsqueeze(-1)
        freq_expanded = self.frequencies.view(1, 1, 1, -1)

        # Calculate arguments: x * freq * 2 * pi
        # Shape: (B, N, C, F)
        args = x_expanded * freq_expanded * 2.0 * np.pi * self.scale

        # Apply sin and cos
        # Shape: (B, N, C, F)
        sin_features = torch.sin(args)
        cos_features = torch.cos(args)

        # Concatenate and flatten
        # Shape: (B, N, C, F, 2) -> (B, N, C * F * 2)
        features = torch.stack([sin_features, cos_features], dim=-1)
        features = features.view(x.shape[0], x.shape[1], -1)

        # Project to output dimension
        return self.projection(features)


class AttentionPooling(nn.Module):
    """
    Aggregates a sequence of vectors into a single vector using a learnable query.
    """

    def __init__(self, model_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, model_dim))
        self.attn = nn.MultiheadAttention(
            embed_dim=model_dim, num_heads=num_heads, batch_first=True, dropout=dropout
        )
        self.norm = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        """
        Args:
            x: (Batch, Seq, Dim)
            key_padding_mask: (Batch, Seq) boolean mask where True indicates padding
        Returns:
            (Batch, Dim)
        """
        batch_size = x.size(0)

        # Expand query for the batch
        # (Batch, 1, Dim)
        query = self.query.expand(batch_size, -1, -1)

        # Attention
        # Output: (Batch, 1, Dim)
        attn_out, _ = self.attn(query, x, x, key_padding_mask=key_padding_mask)

        # Residual + Norm
        out = self.norm(query + self.dropout(attn_out))

        # Squeeze sequence dimension
        return out.squeeze(1)


class SpatiotemporalPointTransformer(nn.Module):
    def __init__(self):
        super().__init__()

        self.model_dim = Config.MODEL_DIM

        # --- 1. Embeddings ---

        # Spatiotemporal Encoding for (x, y, z, t)
        # We use 4 input channels
        self.st_encoder = FourierFeatureEncoding(
            input_dim=4, output_dim=self.model_dim, num_frequencies=16
        )

        # Feature Embedding for (charge, aux)
        # We use 2 input channels
        self.feat_encoder = nn.Sequential(
            nn.Linear(2, self.model_dim),
            nn.LayerNorm(self.model_dim),
            nn.ReLU(),
            nn.Linear(self.model_dim, self.model_dim),
        )

        self.dropout = nn.Dropout(Config.DROPOUT)

        # --- 2. Transformer Backbone ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=Config.NUM_HEADS,
            dim_feedforward=Config.DIM_FEEDFORWARD,
            dropout=Config.DROPOUT,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=Config.NUM_LAYERS
        )

        # --- 3. Pooling ---
        self.pooling = AttentionPooling(
            model_dim=self.model_dim, num_heads=Config.NUM_HEADS, dropout=Config.DROPOUT
        )

        # --- 4. Prediction Head ---
        self.head = nn.Sequential(
            nn.Linear(self.model_dim, self.model_dim),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(self.model_dim, 3),  # Predict vector (x, y, z)
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        """
        Args:
            x: (Batch, N, 6)
               Columns: [x, y, z, time, charge, aux]
        Returns:
            (Batch, 3) Direction vector
        """
        # Create Padding Mask
        # If all features are 0, it's padding.
        # Shape: (Batch, N)
        # True means ignore (padding)
        mask = x.abs().sum(dim=-1) == 0

        # Split inputs
        # coords_time: (B, N, 4) -> x, y, z, t
        coords_time = x[:, :, :4]
        # features: (B, N, 2) -> charge, aux
        features = x[:, :, 4:]

        # Encode Spatiotemporal info
        pos_embedding = self.st_encoder(coords_time)

        # Encode Pulse features
        feat_embedding = self.feat_encoder(features)

        # Combine (Add position to features)
        # This is analogous to adding positional encoding in standard transformers
        embeddings = feat_embedding + pos_embedding
        embeddings = self.dropout(embeddings)

        # Pass through Transformer
        # src_key_padding_mask expects True for padded positions
        encoded = self.transformer(embeddings, src_key_padding_mask=mask)

        # Pooling
        pooled = self.pooling(encoded, key_padding_mask=mask)

        # Prediction
        output = self.head(pooled)

        return output
