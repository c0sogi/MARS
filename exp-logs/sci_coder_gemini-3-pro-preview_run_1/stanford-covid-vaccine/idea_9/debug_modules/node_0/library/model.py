import torch
import torch.nn as nn
import math
from library.config import Config


class SinusoidalPositionalEmbedding(nn.Module):
    """
    Encodes scalar distance values into high-dimensional vectors using sinusoidal functions.
    Used for the 'Pairing Distance' feature.
    """

    def __init__(self, d_model):
        super(SinusoidalPositionalEmbedding, self).__init__()
        self.d_model = d_model

        # Create constant 'pe' matrix with div_term
        # We compute the div_term dynamically in forward to handle arbitrary distance inputs
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        self.register_buffer("div_term", div_term)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Seq_Len) containing signed distances.
        Returns:
            torch.Tensor: Embedded tensor of shape (Batch, Seq_Len, d_model).
        """
        # x shape: [Batch, Seq_Len] -> [Batch, Seq_Len, 1]
        x = x.unsqueeze(-1)

        # Calculate sine and cosine arguments
        # x * div_term -> broadcast to [Batch, Seq_Len, d_model/2]
        args = x * self.div_term

        # Create empty embedding tensor
        pe = torch.zeros(x.size(0), x.size(1), self.d_model, device=x.device)

        # Fill even indices with sin, odd with cos
        pe[:, :, 0::2] = torch.sin(args)
        pe[:, :, 1::2] = torch.cos(args)

        return pe


class ResidualBiGRU(nn.Module):
    """
    A Pre-LayerNorm Residual Bidirectional GRU Block.
    Structure: x = x + Dropout(GRU(LayerNorm(x)))
    """

    def __init__(self, hidden_dim, dropout=0.1):
        super(ResidualBiGRU, self).__init__()
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)  # BiGRU output is 2 * hidden_dim
        self.gru = nn.GRU(
            input_size=hidden_dim * 2,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Seq_Len, 2*Hidden_Dim).
        Returns:
            torch.Tensor: Output tensor of same shape.
        """
        # Pre-LayerNorm
        residual = x
        out = self.layer_norm(x)

        # GRU
        out, _ = self.gru(out)

        # Dropout
        out = self.dropout(out)

        # Residual Connection
        return residual + out


class MultiTaskRNANet(nn.Module):
    """
    Multi-Task Distance-Aware Residual BiGRU Network.

    Features:
    1. Embeddings: Sequence (Masked), Loop Type, Sinusoidal Distance.
    2. Backbone: Stacked Residual BiGRUs.
    3. Heads:
       - Regression Head (Degradation rates)
       - Classification Head (Masked Nucleotide Reconstruction)
    """

    def __init__(self):
        super(MultiTaskRNANet, self).__init__()

        # --- 1. Embeddings ---
        self.seq_embedding = nn.Embedding(
            num_embeddings=Config.VOCAB_SIZE, embedding_dim=Config.EMBED_DIM
        )

        self.loop_embedding = nn.Embedding(
            num_embeddings=Config.LOOP_VOCAB_SIZE, embedding_dim=Config.EMBED_DIM
        )

        self.distance_embedding = SinusoidalPositionalEmbedding(
            d_model=Config.DISTANCE_EMBED_DIM
        )

        # Calculate concatenated input dimension
        # Seq (128) + Loop (128) + Dist (64) = 320
        self.input_dim = Config.EMBED_DIM * 2 + Config.DISTANCE_EMBED_DIM

        # --- 2. Adapter & Backbone ---
        # The backbone operates on 2 * HIDDEN_DIM (BiGRU width)
        self.backbone_dim = Config.HIDDEN_DIM * 2

        self.input_proj = nn.Sequential(
            nn.Linear(self.input_dim, self.backbone_dim),
            nn.LayerNorm(self.backbone_dim),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT),
        )

        # Stacked Residual BiGRU Layers
        self.layers = nn.ModuleList(
            [
                ResidualBiGRU(Config.HIDDEN_DIM, Config.DROPOUT)
                for _ in range(Config.NUM_LAYERS)
            ]
        )

        # --- 3. Heads ---
        # Degradation Head (Regression) -> 5 targets
        self.regression_head = nn.Sequential(
            nn.Linear(self.backbone_dim, self.backbone_dim // 2),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(self.backbone_dim // 2, Config.NUM_TARGETS),
        )

        # Reconstruction Head (Classification) -> 4 bases (A, G, C, U)
        # Note: We output logits for the 4 bases. The MASK token is input only.
        self.reconstruction_head = nn.Sequential(
            nn.Linear(self.backbone_dim, self.backbone_dim // 2),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(self.backbone_dim // 2, 4),
        )

    def forward(self, seq_input, loop_input, dist_input):
        """
        Args:
            seq_input (torch.Tensor): (Batch, Seq_Len) indices.
            loop_input (torch.Tensor): (Batch, Seq_Len) indices.
            dist_input (torch.Tensor): (Batch, Seq_Len) float distances.

        Returns:
            dict: {
                "pred_degradation": (Batch, Seq_Len, 5),
                "pred_reconstruction": (Batch, Seq_Len, 4)
            }
        """
        # 1. Embed Inputs
        emb_seq = self.seq_embedding(seq_input)  # (B, L, 128)
        emb_loop = self.loop_embedding(loop_input)  # (B, L, 128)
        emb_dist = self.distance_embedding(dist_input)  # (B, L, 64)

        # Concatenate features
        x = torch.cat([emb_seq, emb_loop, emb_dist], dim=-1)  # (B, L, 320)

        # Project to backbone dimension
        x = self.input_proj(x)  # (B, L, 512)

        # 2. Pass through Backbone
        for layer in self.layers:
            x = layer(x)

        # 3. Heads
        pred_degradation = self.regression_head(x)  # (B, L, 5)
        pred_reconstruction = self.reconstruction_head(x)  # (B, L, 4)

        return {
            "pred_degradation": pred_degradation,
            "pred_reconstruction": pred_reconstruction,
        }
