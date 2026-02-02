import torch
import torch.nn as nn
import math
from library.config import Config


class SinusoidalPositionalEncoding(nn.Module):
    """
    Encodes scalar distances into a high-dimensional vector using sine and cosine functions.
    Adapted for signed distances (forward/backward pairing).
    """

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

        # Create constant 'div_term' based on the standard Transformer PE formula
        # div_term = 1 / (10000 ^ (2i / d_model))
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        self.register_buffer("div_term", div_term)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (Batch, Seq_Len) containing signed distances.
        Returns:
            Tensor of shape (Batch, Seq_Len, d_model)
        """
        # x shape: [B, L]
        # div_term shape: [d_model/2]

        # Unsqueeze x to [B, L, 1] for broadcasting against div_term
        x_expanded = x.unsqueeze(-1)

        # Calculate arguments for sin and cos
        # [B, L, 1] * [1, 1, d_model/2] -> [B, L, d_model/2]
        args = x_expanded * self.div_term

        # Create empty encoding tensor
        pe = torch.zeros(x.size(0), x.size(1), self.d_model, device=x.device)

        # Fill even indices with sin, odd with cos
        pe[:, :, 0::2] = torch.sin(args)
        pe[:, :, 1::2] = torch.cos(args)

        return pe


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block adapted for 1D sequences.
    Recalibrates channel-wise feature responses by explicitly modelling interdependencies between channels.
    """

    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (Batch, Seq_Len, Channels)
        Returns:
            Tensor of shape (Batch, Seq_Len, Channels)
        """
        b, l, c = x.size()

        # Squeeze: Global Average Pooling
        # Permute to (B, C, L) for pooling -> Output (B, C, 1)
        y = x.permute(0, 2, 1)
        y = self.avg_pool(y).view(b, c)

        # Excitation: MLP to generate channel weights
        y = self.fc(y).view(b, 1, c)

        # Scale: Element-wise multiplication (broadcasting over Seq_Len)
        return x * y


class ResBiGRUBlock(nn.Module):
    """
    Residual Block containing LayerNorm, BiGRU, SE-Attention, and Dropout.
    Uses Pre-LayerNorm configuration: x + Block(Norm(x))
    """

    def __init__(self, hidden_dim, dropout=0.1, se_reduction=16):
        super().__init__()

        self.norm = nn.LayerNorm(hidden_dim)

        # BiGRU: Hidden size is half of input to ensure output dim matches input dim after concatenation
        # Input: hidden_dim -> Output: 2 * (hidden_dim // 2) = hidden_dim
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.se = SEBlock(hidden_dim, reduction=se_reduction)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x

        # Pre-Norm
        out = self.norm(x)

        # BiGRU
        out, _ = self.gru(out)

        # Squeeze-and-Excitation
        out = self.se(out)

        # Dropout
        out = self.dropout(out)

        # Residual Connection
        return residual + out


class RNAModel(nn.Module):
    """
    Channel-Attentive Distance-Aware Residual BiGRU Model.
    """

    def __init__(self):
        super().__init__()

        # 1. Embeddings
        self.seq_embedding = nn.Embedding(Config.VOCAB_SIZE, Config.EMBED_DIM)
        self.loop_embedding = nn.Embedding(Config.LOOP_VOCAB_SIZE, Config.EMBED_DIM)
        self.dist_encoding = SinusoidalPositionalEncoding(Config.EMBED_DIM)

        # 2. Feature Fusion
        # Concatenated dimension: 3 * EMBED_DIM
        input_dim = 3 * Config.EMBED_DIM
        self.feature_projection = nn.Sequential(
            nn.Linear(input_dim, Config.HIDDEN_DIM),
            nn.LayerNorm(Config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT),
        )

        # 3. Backbone (Deep Residual BiGRU with SE)
        self.layers = nn.ModuleList(
            [
                ResBiGRUBlock(
                    hidden_dim=Config.HIDDEN_DIM,
                    dropout=Config.DROPOUT,
                    se_reduction=Config.SE_REDUCTION,
                )
                for _ in range(Config.N_LAYERS)
            ]
        )

        # 4. Output Head
        self.head = nn.Linear(Config.HIDDEN_DIM, Config.NUM_TARGETS)

    def forward(self, sequence, loop_type, pair_dist):
        """
        Args:
            sequence: (Batch, Seq_Len) LongTensor
            loop_type: (Batch, Seq_Len) LongTensor
            pair_dist: (Batch, Seq_Len) FloatTensor
        Returns:
            logits: (Batch, Seq_Len, Num_Targets)
        """
        # Embeddings
        seq_emb = self.seq_embedding(sequence)  # (B, L, E)
        loop_emb = self.loop_embedding(loop_type)  # (B, L, E)
        dist_emb = self.dist_encoding(pair_dist)  # (B, L, E)

        # Concatenate
        x = torch.cat([seq_emb, loop_emb, dist_emb], dim=-1)  # (B, L, 3E)

        # Project to Hidden Dim
        x = self.feature_projection(x)  # (B, L, H)

        # Pass through Residual Blocks
        for layer in self.layers:
            x = layer(x)

        # Output Projection
        logits = self.head(x)  # (B, L, 3)

        return logits
