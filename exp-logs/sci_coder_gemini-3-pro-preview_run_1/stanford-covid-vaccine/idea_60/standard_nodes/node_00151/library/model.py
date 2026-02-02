import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding for signed structural distances.
    Handles continuous/float inputs by computing sin/cos on the fly.
    """

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        # Compute div_term: 1 / (10000 ^ (2i / d_model))
        # shape: (d_model // 2,)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        self.register_buffer("div_term", div_term)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Signed distances of shape (Batch, Seq_Len).
        Returns:
            torch.Tensor: Encodings of shape (Batch, Seq_Len, d_model).
        """
        # x: (B, L) -> (B, L, 1)
        x_expanded = x.unsqueeze(-1)

        # phase: (B, L, d_model/2)
        phase = x_expanded * self.div_term

        # Compute sin and cos
        sin_enc = torch.sin(phase)
        cos_enc = torch.cos(phase)

        # Concatenate to get (B, L, d_model)
        return torch.cat([sin_enc, cos_enc], dim=-1)


class ResidualLSTMBlock(nn.Module):
    """
    Residual Block with BiLSTM.
    Architecture: x -> LN -> BiLSTM -> Dropout -> + -> x
    """

    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_dim)

        # BiLSTM: Output dimension must match hidden_dim.
        # Since it's bidirectional, hidden_size per direction is hidden_dim // 2.
        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim // 2, batch_first=True, bidirectional=True
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x

        # Pre-LayerNorm
        out = self.ln(x)

        # BiLSTM
        out, _ = self.lstm(out)

        # Dropout
        out = self.dropout(out)

        return residual + out


class RNAModel(nn.Module):
    """
    Stabilized Wide-Stream Residual BiLSTM Model.
    """

    def __init__(self, config=Config):
        super().__init__()

        # --- Embeddings ---
        # 1. Sequence (A, G, C, U)
        self.seq_emb = nn.Embedding(4, config.EMBED_SEQ_DIM)

        # 2. Predicted Loop Type (7 types)
        self.loop_emb = nn.Embedding(7, config.EMBED_LOOP_DIM)

        # 3. Structure Distance (Signed Float)
        self.dist_enc = SinusoidalPositionalEncoding(config.EMBED_DIST_DIM)

        # --- Stem ---
        # Projects concatenated inputs to the residual stream width
        self.stem_lstm = nn.LSTM(
            config.TOTAL_INPUT_DIM,
            config.HIDDEN_DIM // 2,
            batch_first=True,
            bidirectional=True,
        )
        # Note: No dropout after stem as per instructions

        # --- Backbone ---
        # Stack of Residual Blocks
        self.blocks = nn.ModuleList(
            [
                ResidualLSTMBlock(config.HIDDEN_DIM, config.DROPOUT)
                for _ in range(config.NUM_LAYERS)
            ]
        )

        # --- Aggregation ---
        # Scalar Mixture of (Stem + 6 Blocks)
        self.num_outputs = config.NUM_LAYERS + 1
        self.mix_weights = nn.Parameter(torch.zeros(self.num_outputs))

        # --- Head ---
        # Shared projection to 3 targets
        self.head = nn.Linear(config.HIDDEN_DIM, 3)

    def forward(self, x_seq, x_loop, x_dist):
        # 1. Embeddings
        e_seq = self.seq_emb(x_seq)  # (B, L, 128)
        e_loop = self.loop_emb(x_loop)  # (B, L, 64)
        e_dist = self.dist_enc(x_dist)  # (B, L, 64)

        # Fusion
        x = torch.cat([e_seq, e_loop, e_dist], dim=-1)  # (B, L, 256)

        # 2. Stem
        x, _ = self.stem_lstm(x)  # (B, L, 384)

        # Store outputs for aggregation
        layer_outputs = [x]

        # 3. Backbone
        for block in self.blocks:
            x = block(x)
            layer_outputs.append(x)

        # 4. Aggregation
        # Stack: (B, L, Hidden, Num_Layers+1)
        stacked = torch.stack(layer_outputs, dim=-1)

        # Compute weights via Softmax for stability
        weights = F.softmax(self.mix_weights, dim=0)  # (Num_Layers+1,)

        # Weighted Sum
        # Broadcast weights to (1, 1, 1, Num_Layers+1)
        aggregated = torch.sum(
            stacked * weights.view(1, 1, 1, -1), dim=-1
        )  # (B, L, Hidden)

        # 5. Head
        logits = self.head(aggregated)  # (B, L, 3)

        return logits
