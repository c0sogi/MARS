import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class ScalarMixture(nn.Module):
    """
    Computes a learnable weighted sum of a list of tensors.
    Used to aggregate features from different depths of the network (Stem + Blocks).
    """

    def __init__(self, num_layers):
        super(ScalarMixture, self).__init__()
        self.num_layers = num_layers
        # Initialize weights to be equal (zeros -> softmax -> uniform)
        self.weights = nn.Parameter(torch.zeros(num_layers))

    def forward(self, tensors):
        """
        Args:
            tensors (list of torch.Tensor): List of N tensors, each shape (Batch, Seq, Hidden).
        Returns:
            torch.Tensor: Weighted sum, shape (Batch, Seq, Hidden).
        """
        # Stack tensors: (Batch, N, Seq, Hidden)
        stacked = torch.stack(tensors, dim=1)

        # Compute normalized weights via Softmax
        w = F.softmax(self.weights, dim=0)

        # Reshape for broadcasting: (1, N, 1, 1)
        w = w.view(1, self.num_layers, 1, 1)

        # Weighted sum along the layer dimension
        weighted_sum = torch.sum(stacked * w, dim=1)

        return weighted_sum


class ResidualBiLSTMBlock(nn.Module):
    """
    A Residual Block containing a Pre-LayerNorm BiLSTM.
    Structure: Input -> LN -> BiLSTM -> Dropout -> Residual Add
    """

    def __init__(self, hidden_dim, dropout):
        super(ResidualBiLSTMBlock, self).__init__()
        self.ln = nn.LayerNorm(hidden_dim)

        # BiLSTM maintaining the width
        # Hidden size is half of hidden_dim because it's bidirectional
        self.bilstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x

        # Pre-LayerNorm
        out = self.ln(x)

        # BiLSTM
        out, _ = self.bilstm(out)

        # Dropout
        out = self.dropout(out)

        # Residual Connection
        return residual + out


class RNA_Net(nn.Module):
    """
    Deeply-Supervised Wide-Stream Residual BiLSTM Network.
    """

    def __init__(self):
        super(RNA_Net, self).__init__()

        # ==============================
        # 1. Input Embeddings
        # ==============================
        # Sequence: A, G, C, U (4 tokens)
        self.seq_embed = nn.Embedding(4, Config.EMBED_DIM_SEQ)

        # Loop Type: 7 tokens (S, M, I, B, H, E, X)
        self.loop_embed = nn.Embedding(7, Config.EMBED_DIM_LOOP)

        # Distance: Pre-computed sinusoidal encoding (Config.EMBED_DIM_DIST)
        # Total concatenated input dimension
        input_dim = Config.EMBED_DIM_SEQ + Config.EMBED_DIM_LOOP + Config.EMBED_DIM_DIST

        # ==============================
        # 2. High-Fidelity Recurrent Stem
        # ==============================
        # Projects concatenated input to the residual stream width (HIDDEN_DIM)
        # No dropout applied to the stem output
        self.stem = nn.LSTM(
            input_size=input_dim,
            hidden_size=Config.HIDDEN_DIM // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # ==============================
        # 3. Backbone: Residual Blocks
        # ==============================
        self.blocks = nn.ModuleList(
            [
                ResidualBiLSTMBlock(Config.HIDDEN_DIM, Config.DROPOUT)
                for _ in range(Config.NUM_LAYERS)
            ]
        )

        # ==============================
        # 4. Shared Readout Head
        # ==============================
        # Maps hidden state to targets (reactivity, deg_Mg_pH10, deg_Mg_50C)
        self.head = nn.Linear(Config.HIDDEN_DIM, len(Config.TARGET_COLS))

        # ==============================
        # 5. Aggregation: Scalar Mixture
        # ==============================
        # Aggregates Stem + 6 Blocks (Total 7 layers)
        self.mixture = ScalarMixture(num_layers=1 + Config.NUM_LAYERS)

    def forward(self, seq, loop, dist):
        """
        Args:
            seq (torch.LongTensor): (Batch, Seq_Len)
            loop (torch.LongTensor): (Batch, Seq_Len)
            dist (torch.FloatTensor): (Batch, Seq_Len, Embed_Dim_Dist)

        Returns:
            main_pred (torch.Tensor): Final prediction (Batch, Seq_Len, 3)
            layer_preds (list of torch.Tensor): Predictions from each layer for deep supervision
        """
        # 1. Embed Inputs
        emb_seq = self.seq_embed(seq)  # (B, L, 128)
        emb_loop = self.loop_embed(loop)  # (B, L, 64)

        # 2. Fuse (Concatenate)
        # dist is already (B, L, 64)
        x = torch.cat([emb_seq, emb_loop, dist], dim=-1)  # (B, L, 256)

        # 3. Stem Processing
        x, _ = self.stem(x)  # (B, L, 384)

        # Collect outputs for Deep Supervision and Mixture
        # List starts with Stem output
        layer_outputs = [x]

        # 4. Backbone Processing
        for block in self.blocks:
            x = block(x)
            layer_outputs.append(x)

        # 5. Scalar Mixture Aggregation
        h_final = self.mixture(layer_outputs)  # (B, L, 384)

        # 6. Main Prediction
        main_pred = self.head(h_final)  # (B, L, 3)

        return main_pred
