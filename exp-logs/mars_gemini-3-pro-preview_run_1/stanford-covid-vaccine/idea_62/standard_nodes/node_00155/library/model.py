import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class ScalarMixture(nn.Module):
    """
    Computes a learnable weighted sum of a list of tensors.
    Uses softmax normalization on weights to ensure stability.
    """

    def __init__(self, n_layers):
        super(ScalarMixture, self).__init__()
        self.n_layers = n_layers
        # Initialize weights to 0 so that initial contribution is uniform
        self.weights = nn.Parameter(torch.zeros(n_layers))

    def forward(self, tensors):
        """
        Args:
            tensors: List of tensors, each of shape (batch, seq_len, hidden_dim)
        Returns:
            Weighted sum tensor of shape (batch, seq_len, hidden_dim)
        """
        # Stack tensors along a new dimension: (batch, seq, hidden, n_layers)
        stacked = torch.stack(tensors, dim=-1)

        # Compute normalized weights
        norm_weights = F.softmax(self.weights, dim=0)  # (n_layers,)

        # Weighted sum
        # Broadcast weights: (1, 1, 1, n_layers)
        weighted_sum = torch.sum(stacked * norm_weights.view(1, 1, 1, -1), dim=-1)

        return weighted_sum


class ResidualBiLSTMBlock(nn.Module):
    """
    Residual Block with Pre-LayerNorm BiLSTM.
    Cite solution_lesson_node_00135 (Pre-LayerNorm)
    Cite solution_lesson_node_00150 (LSTM vs GRU)
    """

    def __init__(self, hidden_dim, dropout):
        super(ResidualBiLSTMBlock, self).__init__()

        self.norm = nn.LayerNorm(hidden_dim)

        # BiLSTM: Input hidden_dim -> Output hidden_dim (2 * hidden_dim//2)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            batch_first=True,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: Input tensor (batch, seq_len, hidden_dim)
        """
        residual = x

        # Pre-LayerNorm
        x_norm = self.norm(x)

        # BiLSTM
        rnn_out, _ = self.lstm(x_norm)

        # Dropout
        out = self.dropout(rnn_out)

        # Residual Connection
        return residual + out


class RNA_Model(nn.Module):
    """
    Wide-Stream Residual BiLSTM Model.
    Cite solution_lesson_node_00154 (Remove Sequence Gating)
    Cite solution_lesson_node_00150 (LSTM Backbone)
    """

    def __init__(self):
        super(RNA_Model, self).__init__()

        # 1. Embeddings
        # Atomic Sequence
        self.seq_embed = nn.Embedding(4, Config.EMBED_DIM_SEQ)

        # Predicted Loop Type
        self.loop_embed = nn.Embedding(7, Config.EMBED_DIM_LOOP)

        # Distance Features are passed as pre-computed sinusoidal encodings (float)
        # Total Input Dimension
        input_dim = Config.EMBED_DIM_SEQ + Config.EMBED_DIM_LOOP + Config.EMBED_DIM_DIST

        # 2. BiLSTM Stem
        # Projects concatenated inputs to the residual stream width
        # Cite solution_lesson_node_00109 (No Stem Dropout)
        self.stem_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=Config.HIDDEN_DIM // 2,
            batch_first=True,
            bidirectional=True,
        )

        # 3. Backbone: Stack of Residual BiLSTM Blocks
        self.layers = nn.ModuleList(
            [
                ResidualBiLSTMBlock(
                    hidden_dim=Config.HIDDEN_DIM,
                    dropout=Config.DROPOUT,
                )
                for _ in range(Config.NUM_LAYERS)
            ]
        )

        # 4. Scalar Mixture Aggregation
        # Aggregates outputs from Stem + all Layers
        self.mixture = ScalarMixture(Config.NUM_LAYERS + 1)

        # 5. Output Head
        self.head = nn.Linear(Config.HIDDEN_DIM, Config.OUTPUT_DIM)

    def forward(self, seq, loop, dist):
        """
        Args:
            seq: (batch, seq_len) LongTensor
            loop: (batch, seq_len) LongTensor
            dist: (batch, seq_len, dist_dim) FloatTensor
        """
        # Embeddings
        emb_seq = self.seq_embed(seq)  # (B, L, 128)
        emb_loop = self.loop_embed(loop)  # (B, L, 64)

        # Concatenate (Fusion)
        # dist is (B, L, 64)
        x = torch.cat([emb_seq, emb_loop, dist], dim=-1)  # (B, L, 256)

        # Stem
        stem_out, _ = self.stem_lstm(x)  # (B, L, 384)

        # Backbone
        layer_outputs = [stem_out]
        current_x = stem_out

        for layer in self.layers:
            current_x = layer(current_x)
            layer_outputs.append(current_x)

        # Aggregation
        agg_out = self.mixture(layer_outputs)  # (B, L, 384)

        # Head
        logits = self.head(agg_out)  # (B, L, 3)

        return logits
