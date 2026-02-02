import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from library.config import Config


class SinusoidalEncoding(nn.Module):
    """
    Generates sinusoidal encodings for signed structural distances.
    Preserves sign information to distinguish upstream/downstream dependencies.
    """

    def __init__(self, dim, max_len=10000.0):
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        # Create the division term for the frequencies
        # div_term = 1 / (max_len ^ (2i / dim))
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * -(math.log(max_len) / dim)
        )
        self.register_buffer("div_term", div_term)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (Batch, Seq_Len) containing signed float distances.
        Returns:
            Tensor of shape (Batch, Seq_Len, Dim)
        """
        # x shape: (B, L) -> (B, L, 1)
        x = x.unsqueeze(-1)

        # div_term shape: (Dim/2,)
        # Argument shape: (B, L, Dim/2)
        args = x * self.div_term

        # Create embedding: (B, L, Dim)
        # Interleave sin and cos
        pe = torch.zeros(x.shape[0], x.shape[1], self.dim, device=x.device)
        pe[:, :, 0::2] = torch.sin(args)
        pe[:, :, 1::2] = torch.cos(args)

        return pe


class ResidualBiLSTMBlock(nn.Module):
    """
    Standard Pre-LayerNorm Residual BiLSTM Block.
    Structure: Input -> LN -> BiLSTM -> Dropout -> Residual + Input
    Cite Lesson 135: Superiority of Pre-LayerNorm Residual Blocks.
    """

    def __init__(self, width, dropout):
        super().__init__()
        self.layer_norm = nn.LayerNorm(width)

        # Standard BiLSTM: Hidden size = Width // 2.
        # Output size will be Width (e.g., 192 * 2 = 384).
        self.lstm = nn.LSTM(
            input_size=width,
            hidden_size=width // 2,
            batch_first=True,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x

        # 1. Pre-Norm
        out = self.layer_norm(x)

        # 2. BiLSTM
        out, _ = self.lstm(out)

        # 3. Dropout
        out = self.dropout(out)

        # 4. Residual
        return residual + out


class ScalarMixture(nn.Module):
    """
    Computes a learnable weighted sum of a list of tensors.
    """

    def __init__(self, num_layers):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(num_layers))

    def forward(self, tensors):
        """
        Args:
            tensors: List of K tensors, each shape (B, L, D)
        Returns:
            Weighted sum tensor of shape (B, L, D)
        """
        # Normalize weights using softmax
        norm_weights = F.softmax(self.weights, dim=0)

        # Compute weighted sum
        # Stack tensors: (K, B, L, D)
        stacked = torch.stack(tensors, dim=0)

        # Broadcast weights: (K, 1, 1, 1)
        w_expanded = norm_weights.view(-1, 1, 1, 1)

        weighted_sum = torch.sum(stacked * w_expanded, dim=0)
        return weighted_sum


class RNAModel(nn.Module):
    def __init__(self):
        super().__init__()

        # =====================================================================
        # 1. Embeddings
        # =====================================================================
        self.seq_embedding = nn.Embedding(4, Config.INPUT_DIM_SEQ)  # A, G, C, U
        self.loop_embedding = nn.Embedding(7, Config.INPUT_DIM_LOOP)  # Loop types
        self.struct_encoding = SinusoidalEncoding(Config.INPUT_DIM_STRUCT)

        total_input_dim = (
            Config.INPUT_DIM_SEQ + Config.INPUT_DIM_LOOP + Config.INPUT_DIM_STRUCT
        )

        # =====================================================================
        # 2. Stem
        # =====================================================================
        # Project input to residual stream width via BiLSTM
        # To get output dim = STREAM_WIDTH, hidden_size must be STREAM_WIDTH // 2
        self.stem_lstm = nn.LSTM(
            input_size=total_input_dim,
            hidden_size=Config.STREAM_WIDTH // 2,
            batch_first=True,
            bidirectional=True,
        )

        # =====================================================================
        # 3. Backbone (Projected Blocks)
        # =====================================================================
        self.blocks = nn.ModuleList(
            [
                ProjectedBiLSTMBlock(width=Config.STREAM_WIDTH, dropout=Config.DROPOUT)
                for _ in range(Config.NUM_LAYERS)
            ]
        )

        # =====================================================================
        # 4. Aggregation & Head
        # =====================================================================
        # We aggregate outputs from the Stem + 6 Blocks (Total 7 layers)
        self.mixture = ScalarMixture(num_layers=Config.NUM_LAYERS + 1)

        self.head = nn.Linear(Config.STREAM_WIDTH, 3)  # 3 targets

        # =====================================================================
        # 5. Initialization
        # =====================================================================
        self._init_weights()

    def _init_weights(self):
        """
        Orthogonal initialization for Recurrent weights.
        Xavier/Kaiming for others implicitly or explicitly if needed.
        """
        for name, param in self.named_parameters():
            if "lstm" in name and "weight_hh" in name:
                nn.init.orthogonal_(param, gain=1.0)
            elif "lstm" in name and "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, seq, loop, struct):
        # 1. Construct Inputs
        emb_seq = self.seq_embedding(seq)  # (B, L, 128)
        emb_loop = self.loop_embedding(loop)  # (B, L, 64)
        emb_struct = self.struct_encoding(struct)  # (B, L, 64)

        # Concatenate: (B, L, 256)
        x = torch.cat([emb_seq, emb_loop, emb_struct], dim=-1)

        # 2. Stem
        # No dropout on stem output
        x, _ = self.stem_lstm(x)  # (B, L, 384)

        layer_outputs = [x]

        # 3. Backbone
        for block in self.blocks:
            x = block(x)
            layer_outputs.append(x)

        # 4. Aggregation
        # Mix stem output and all block outputs
        x_agg = self.mixture(layer_outputs)

        # 5. Head
        logits = self.head(x_agg)  # (B, L, 3)

        return logits
