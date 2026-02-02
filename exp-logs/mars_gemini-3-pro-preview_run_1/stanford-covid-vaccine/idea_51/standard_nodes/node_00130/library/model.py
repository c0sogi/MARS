import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from library.config import Config


class SinusoidalSignedPositionalEncoding(nn.Module):
    """
    Sinusoidal encoding that handles signed float distances.
    Preserves sign information via the phase of sine/cosine functions.
    """

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        # Ensure d_model is even for simple sin/cos splitting
        if d_model % 2 != 0:
            raise ValueError("SinusoidalSignedPositionalEncoding d_model must be even.")

        # div_term: 10000^(-2i/d_model)
        # We compute this once and register it as a buffer
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        self.register_buffer("div_term", div_term)

    def forward(self, x):
        """
        Args:
            x: (Batch, Seq_Len) tensor of signed float distances.
        Returns:
            (Batch, Seq_Len, d_model) positional encoding.
        """
        # x shape: (B, L)
        # div_term shape: (d_model/2,)

        # Unsqueeze x to (B, L, 1) to broadcast against div_term
        x_expanded = x.unsqueeze(-1)

        # Compute phase: (B, L, d_model/2)
        # Note: Since x can be negative, the phase will be negative,
        # preserving directionality in sin/cos space.
        phase = x_expanded * self.div_term

        # Create output tensor
        pe = torch.zeros(x.shape[0], x.shape[1], self.d_model, device=x.device)

        # Assign sin to even indices, cos to odd indices
        pe[..., 0::2] = torch.sin(phase)
        pe[..., 1::2] = torch.cos(phase)

        return pe


class ResidualBlock(nn.Module):
    """
    A Pre-LN BiLSTM block.
    Uses a wide residual stream and applies dropout.
    Removed LayerDrop (Cite Lesson 00129).
    """

    def __init__(self, hidden_dim, dropout):
        super().__init__()

        # Pre-LayerNorm configuration
        self.ln = nn.LayerNorm(hidden_dim)

        # Wide-Stream BiLSTM (Cite Lesson 00112)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            batch_first=True,
            bidirectional=True,
        )

        # Inter-layer Dropout (Cite Lesson 00076)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x

        # Pre-LN
        out = self.ln(x)

        # Transformation
        out, _ = self.lstm(out)
        out = self.dropout(out)

        # Standard Residual Connection
        return residual + out


class RNAModel(nn.Module):
    """
    Stabilized Wide-Stream BiLSTM Model.
    """

    def __init__(self, config=Config):
        super().__init__()

        # 1. Heterogeneous Feature Embeddings
        self.seq_embed = nn.Embedding(4, config.EMBED_DIM)
        self.loop_embed = nn.Embedding(7, config.LOOP_DIM)
        self.pair_embed = SinusoidalSignedPositionalEncoding(config.PAIR_DIM)

        input_dim = config.EMBED_DIM + config.LOOP_DIM + config.PAIR_DIM

        # 2. High-Fidelity Stem (No Dropout)
        # Projects concatenated inputs to the wide stream capacity
        # Using LSTM (Cite Lesson 00112)
        # No Dropout here (Cite Lesson 00109)
        self.stem = nn.LSTM(
            input_size=input_dim,
            hidden_size=config.HIDDEN_DIM // 2,
            batch_first=True,
            bidirectional=True,
        )

        # 3. Backbone
        self.blocks = nn.ModuleList()
        for _ in range(config.N_LAYERS):
            block = ResidualBlock(
                hidden_dim=config.HIDDEN_DIM,
                dropout=config.DROPOUT,
            )
            self.blocks.append(block)

        # 4. Global Static Aggregation (Scalar Mixture)
        # Learnable weights for Stem + N Layers
        self.mix_weights = nn.Parameter(torch.zeros(n_layers + 1))

        # 5. Output Head
        # Projects to the 3 scored targets
        self.head = nn.Linear(config.HIDDEN_DIM, 3)

    def forward(self, sequence, loop_type, pair_dist):
        # Embed inputs
        emb_seq = self.seq_embed(sequence)  # (B, L, 128)
        emb_loop = self.loop_embed(loop_type)  # (B, L, 64)
        emb_pair = self.pair_embed(pair_dist)  # (B, L, 64)

        # Concatenate features
        x = torch.cat([emb_seq, emb_loop, emb_pair], dim=-1)  # (B, L, 256)

        # Pass through Stem
        x, _ = self.stem(x)  # (B, L, 512)

        # Store outputs for mixture (Index 0 is Stem)
        layer_outputs = [x]

        # Pass through Backbone Blocks
        for block in self.blocks:
            x = block(x)
            layer_outputs.append(x)

        # Scalar Mixture Aggregation
        # Stack outputs: (B, L, H, N+1)
        stacked_outputs = torch.stack(layer_outputs, dim=-1)

        # Normalize weights via Softmax to ensure stability
        norm_weights = F.softmax(self.mix_weights, dim=0)

        # Weighted Sum: sum(output_i * weight_i)
        # Weights broadcast automatically over B, L, H
        weighted_out = torch.sum(stacked_outputs * norm_weights, dim=-1)  # (B, L, H)

        # Final Projection
        logits = self.head(weighted_out)  # (B, L, 3)

        return logits
