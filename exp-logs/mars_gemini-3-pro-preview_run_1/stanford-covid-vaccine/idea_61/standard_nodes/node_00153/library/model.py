import torch
import torch.nn as nn
import math
from library.config import Config


class SinusoidalSignedPositionalEncoding(nn.Module):
    """
    Implements fixed sinusoidal encodings for signed integers (pairing distances).
    Preserves sign information via the odd property of the sine function.
    """

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        # Create a scalar for the division term to keep it on the correct device
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        self.register_buffer("div_term", div_term)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (Batch, Seq_Len) containing signed integer distances.
        Returns:
            Tensor of shape (Batch, Seq_Len, d_model)
        """
        # x shape: [Batch, Seq_Len, 1] or [Batch, Seq_Len]
        if x.dim() == 3:
            x = x.squeeze(-1)

        # Create position encodings
        # pe shape: [Batch, Seq_Len, d_model]
        pe = torch.zeros(x.size(0), x.size(1), self.d_model, device=x.device)

        # Calculate arguments for sin/cos
        # x.unsqueeze(-1) -> [Batch, Seq_Len, 1]
        # self.div_term -> [d_model/2]
        # arg -> [Batch, Seq_Len, d_model/2]
        arg = x.unsqueeze(-1).float() * self.div_term

        # Assign sin to even indices
        pe[..., 0::2] = torch.sin(arg)
        # Assign cos to odd indices
        pe[..., 1::2] = torch.cos(arg)

        return pe


class HeterogeneousEmbeddings(nn.Module):
    """
    Embeds sequence, loop type, and pairing distance into a unified vector.
    """

    def __init__(self):
        super().__init__()

        # 1. Atomic Sequence Embedding (A, G, C, U)
        self.seq_embedding = nn.Embedding(4, Config.EMB_SEQ_DIM)

        # 2. Predicted Loop Type Embedding (S, M, I, B, H, E, X)
        self.loop_embedding = nn.Embedding(7, Config.EMB_LOOP_DIM)

        # 3. Signed Sinusoidal Pairing Distance
        self.pair_embedding = SinusoidalSignedPositionalEncoding(Config.EMB_PAIR_DIM)

    def forward(self, inputs):
        """
        Args:
            inputs: (Batch, Seq_Len, 3)
                Slice 0: Sequence indices
                Slice 1: Loop indices
                Slice 2: Distance integers
        """
        seq_idx = inputs[:, :, 0]
        loop_idx = inputs[:, :, 1]
        dists = inputs[:, :, 2]

        emb_seq = self.seq_embedding(seq_idx)  # (B, L, 128)
        emb_loop = self.loop_embedding(loop_idx)  # (B, L, 64)
        emb_pair = self.pair_embedding(dists)  # (B, L, 64)

        # Concatenate: 128 + 64 + 64 = 256
        return torch.cat([emb_seq, emb_loop, emb_pair], dim=-1)


class PreNormBiLSTMBlock(nn.Module):
    """
    A Residual Block using Pre-LayerNorm and BiLSTM.
    Maintains the residual stream width.
    Structure: Input -> LN -> BiLSTM -> Dropout -> + Input
    """

    def __init__(self, hidden_dim, dropout):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # To output `hidden_dim` (512) from a bidirectional LSTM,
        # the internal hidden size must be hidden_dim // 2 (256).
        self.bilstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = self.layer_norm(x)
        out, _ = self.bilstm(out)
        out = self.dropout(out)
        return residual + out


class ScalarMixture(nn.Module):
    """
    Computes a learnable weighted sum of a list of tensors.
    """

    def __init__(self, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.weights = nn.Parameter(torch.zeros(num_layers))

    def forward(self, tensors):
        """
        Args:
            tensors: List of tensors, each (Batch, Seq_Len, Hidden_Dim)
        """
        # Stack tensors: (Batch, Seq_Len, Hidden_Dim, Num_Layers)
        stacked = torch.stack(tensors, dim=-1)

        # Compute softmax weights
        norm_weights = torch.softmax(self.weights, dim=0)

        # Weighted sum
        # (B, L, H, N) * (N) -> (B, L, H)
        weighted_sum = torch.sum(stacked * norm_weights, dim=-1)

        return weighted_sum


class StabilizedWideBiLSTM(nn.Module):
    """
    The main model architecture implementing the 'Stabilized High-Capacity Wide-Stream BiLSTM' strategy.
    """

    def __init__(self):
        super().__init__()

        # 1. Embeddings
        self.embeddings = HeterogeneousEmbeddings()

        # 2. BiLSTM Stem
        # Projects fused input (256) to residual stream width (512)
        # No dropout here.
        self.stem = nn.LSTM(
            input_size=Config.INPUT_DIM,
            hidden_size=Config.HIDDEN_DIM // 2,
            batch_first=True,
            bidirectional=True,
        )

        # 3. Backbone
        self.blocks = nn.ModuleList(
            [
                PreNormBiLSTMBlock(Config.HIDDEN_DIM, Config.DROPOUT)
                for _ in range(Config.NUM_LAYERS)
            ]
        )

        # 4. Aggregation
        # We aggregate outputs from the Stem + 6 Blocks (Total 7)
        self.mixture = ScalarMixture(num_layers=1 + Config.NUM_LAYERS)

        # 5. Output Head
        self.head = nn.Linear(Config.HIDDEN_DIM, Config.NUM_TARGETS)

    def forward(self, inputs):
        """
        Args:
            inputs: (Batch, Seq_Len, 3)
        Returns:
            (Batch, Seq_Len, Num_Targets)
        """
        # Embed
        x = self.embeddings(inputs)

        # Stem
        x, _ = self.stem(x)

        # Collect layer outputs for mixture
        layer_outputs = [x]

        # Backbone
        for block in self.blocks:
            x = block(x)
            layer_outputs.append(x)

        # Aggregate
        x_aggregated = self.mixture(layer_outputs)

        # Project
        logits = self.head(x_aggregated)

        return logits
