import torch
import torch.nn as nn
import math
from library.config import Config


class SinusoidalPositionalEncoding(nn.Module):
    """
    Encodes scalar distances into dense vectors using sinusoidal functions.
    Adapted from standard Transformer Positional Encoding but applied to
    structural distances (values) rather than sequence indices.
    """

    def __init__(self, d_model):
        super(SinusoidalPositionalEncoding, self).__init__()
        self.d_model = d_model

        # Precompute the div_term for the sinusoidal functions
        # formula: exp(arange(0, d, 2) * -(log(10000.0) / d))
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        self.register_buffer("div_term", div_term)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (Batch, Seq_Len) containing scalar distances.
        Returns:
            Tensor of shape (Batch, Seq_Len, d_model)
        """
        # Unsqueeze to broadcast: (B, L) -> (B, L, 1)
        x_expanded = x.unsqueeze(-1)

        # Calculate phase: (B, L, 1) * (d_model/2,) -> (B, L, d_model/2)
        phase = x_expanded * self.div_term

        # Compute Sin and Cos
        pe_sin = torch.sin(phase)
        pe_cos = torch.cos(phase)

        # Concatenate to get full d_model dimension: (B, L, d_model)
        return torch.cat([pe_sin, pe_cos], dim=-1)


class ResidualBiGRU(nn.Module):
    """
    A Bidirectional GRU layer with a residual connection and LayerNorm.
    Forward pass: Output = LayerNorm(Input + Dropout(BiGRU(Input)))
    """

    def __init__(self, input_dim, hidden_dim, dropout=0.0):
        super(ResidualBiGRU, self).__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(input_dim)

        # For a residual connection x + f(x), the input and output dimensions must match.
        # BiGRU output dimension is 2 * hidden_dim.
        if input_dim != 2 * hidden_dim:
            raise ValueError(
                f"Input dim ({input_dim}) must match BiGRU output dim (2*{hidden_dim}) for residual connection."
            )

    def forward(self, x):
        # x: (Batch, Seq, Input_Dim)
        # out: (Batch, Seq, 2*Hidden_Dim)
        out, _ = self.gru(x)
        out = self.dropout(out)

        # Residual connection + Normalization
        return self.norm(x + out)


class RNAModel(nn.Module):
    """
    Distance-Aware Residual BiGRU Network.
    Combines Sequence, Loop, and Structural Distance features into a deep residual RNN.
    """

    def __init__(self, config=Config):
        super(RNAModel, self).__init__()

        # ----------------------------------------------------------------------
        # 1. Feature Embeddings
        # ----------------------------------------------------------------------
        self.seq_embedding = nn.Embedding(config.VOCAB_SIZE_SEQ, config.EMBED_DIM)
        self.loop_embedding = nn.Embedding(config.VOCAB_SIZE_LOOP, config.EMBED_DIM)
        self.dist_encoding = SinusoidalPositionalEncoding(config.DISTANCE_EMBED_DIM)

        # Calculate total input dimension
        # e.g., 128 (Seq) + 128 (Loop) + 64 (Dist) = 320
        self.input_dim = (2 * config.EMBED_DIM) + config.DISTANCE_EMBED_DIM

        # ----------------------------------------------------------------------
        # 2. Backbone
        # ----------------------------------------------------------------------
        # Layer 0: Adaptation Layer
        # Transforms input features (320) to the backbone width (2 * 256 = 512)
        # No residual here as dimensions change.
        self.initial_gru = nn.GRU(
            input_size=self.input_dim,
            hidden_size=config.HIDDEN_DIM,
            batch_first=True,
            bidirectional=True,
        )
        self.initial_dropout = nn.Dropout(config.DROPOUT)
        self.initial_norm = nn.LayerNorm(config.HIDDEN_DIM * 2)

        # Layers 1..N: Deep Residual Stack
        # Input and Output are both 512 (2 * HIDDEN_DIM)
        self.residual_layers = nn.ModuleList(
            [
                ResidualBiGRU(
                    input_dim=config.HIDDEN_DIM * 2,
                    hidden_dim=config.HIDDEN_DIM,
                    dropout=config.DROPOUT,
                )
                for _ in range(config.N_LAYERS - 1)
            ]
        )

        # ----------------------------------------------------------------------
        # 3. Prediction Head
        # ----------------------------------------------------------------------
        self.head = nn.Linear(config.HIDDEN_DIM * 2, config.NUM_TARGETS)

    def forward(self, sequence, loop, distance):
        """
        Args:
            sequence: (Batch, Seq_Len) - Integer indices
            loop: (Batch, Seq_Len) - Integer indices
            distance: (Batch, Seq_Len) - Float scalar distances
        Returns:
            logits: (Batch, Seq_Len, 5)
        """
        # 1. Embed Inputs
        seq_emb = self.seq_embedding(sequence)  # (B, L, 128)
        loop_emb = self.loop_embedding(loop)  # (B, L, 128)
        dist_emb = self.dist_encoding(distance)  # (B, L, 64)

        # 2. Concatenate
        x = torch.cat([seq_emb, loop_emb, dist_emb], dim=-1)  # (B, L, 320)

        # 3. Initial Adaptation Layer
        x, _ = self.initial_gru(x)  # (B, L, 512)
        x = self.initial_dropout(x)
        x = self.initial_norm(x)

        # 4. Residual Backbone
        for layer in self.residual_layers:
            x = layer(x)  # (B, L, 512)

        # 5. Output Head
        logits = self.head(x)  # (B, L, 5)

        return logits
