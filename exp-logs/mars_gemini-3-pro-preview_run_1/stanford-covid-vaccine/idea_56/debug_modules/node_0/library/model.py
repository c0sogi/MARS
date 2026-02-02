import torch
import torch.nn as nn
import math
from library.config import Config


class SinusoidalDistanceEncoder(nn.Module):
    """
    Encodes scalar distance values into high-dimensional vectors using
    fixed sinusoidal functions (sine and cosine).
    Preserves the sign of the distance to distinguish upstream/downstream relationships.
    """

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        # Compute the division term for the sinusoidal frequencies
        # pe(x) = [sin(x * w_k), cos(x * w_k)]
        # w_k = 1 / 10000^(2k/d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        self.register_buffer("div_term", div_term)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Signed distances of shape (Batch, Seq_Len).
        Returns:
            torch.Tensor: Embeddings of shape (Batch, Seq_Len, d_model).
        """
        # x.unsqueeze(-1) -> (B, L, 1)
        # div_term -> (d_model/2)
        # argument -> (B, L, d_model/2)
        x_expanded = x.unsqueeze(-1)
        argument = x_expanded * self.div_term

        # Initialize embedding tensor
        pe = torch.zeros(*x.shape, self.d_model, device=x.device)

        # Apply sin to even indices and cos to odd indices
        pe[..., 0::2] = torch.sin(argument)
        pe[..., 1::2] = torch.cos(argument)

        return pe


class ChannelScaledResidualBlock(nn.Module):
    """
    A residual block containing a Pre-LayerNorm BiGRU with Channel-Wise Residual Scaling.

    Structure:
        y = x + Scale * Dropout(BiGRU(LayerNorm(x)))

    The 'Scale' is a learnable diagonal matrix (vector) initialized to 1.0, allowing
    the network to dampen or amplify specific feature channels in the residual branch.
    """

    def __init__(self, hidden_dim, dropout_p):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # BiGRU: Hidden size is halved so that concatenation (Forward+Backward) matches hidden_dim
        self.bigru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            batch_first=True,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(dropout_p)

        # Learnable channel-wise scaling vector, initialized to 1.0 (Identity)
        self.scale = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, x):
        residual = x

        # Pre-LayerNorm
        out = self.layer_norm(x)

        # BiGRU
        out, _ = self.bigru(out)

        # Dropout
        out = self.dropout(out)

        # Channel Scaling (Element-wise multiplication)
        # Broadcasting scale (Hidden,) to (Batch, Seq, Hidden)
        out = out * self.scale

        # Residual Connection
        return residual + out


class ScalarMixture(nn.Module):
    """
    Aggregates outputs from multiple layers using a learnable weighted sum.
    Uses Softmax to ensure weights are normalized.
    """

    def __init__(self, num_layers):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(num_layers))

    def forward(self, layers_output):
        """
        Args:
            layers_output (list of torch.Tensor): List of tensors, each (Batch, Seq, Hidden).
        Returns:
            torch.Tensor: Weighted sum tensor of shape (Batch, Seq, Hidden).
        """
        # Stack layers: (Num_Layers, Batch, Seq, Hidden)
        stacked = torch.stack(layers_output, dim=0)

        # Compute normalized weights
        probs = torch.softmax(self.weights, dim=0)

        # Reshape for broadcasting: (Num_Layers, 1, 1, 1)
        probs = probs.view(-1, 1, 1, 1)

        # Weighted sum
        weighted_sum = (stacked * probs).sum(dim=0)
        return weighted_sum


class RNAModel(nn.Module):
    """
    Channel-Scaled Wide-Stream Residual BiGRU Model.

    Architecture:
    1. Heterogeneous Embeddings (Seq, Loop, Distance)
    2. High-Fidelity BiGRU Stem (No Dropout)
    3. Deep Backbone of Channel-Scaled Residual Blocks
    4. Scalar Mixture Aggregation
    5. Regression Head
    """

    def __init__(self):
        super().__init__()

        # ------------------------------------------------------------------
        # 1. Embeddings
        # ------------------------------------------------------------------
        # Sequence: 4 tokens (A, G, C, U) -> 128 dim
        # Vocab size 5 to account for potential padding index 0
        self.seq_embed = nn.Embedding(5, Config.EMBED_DIM_SEQ, padding_idx=0)

        # Loop Type: 7 tokens -> 64 dim
        # Vocab size 8 to account for potential padding index 0
        self.loop_embed = nn.Embedding(8, Config.EMBED_DIM_LOOP, padding_idx=0)

        # Distance: Signed Sinusoidal -> 64 dim
        self.dist_embed = SinusoidalDistanceEncoder(Config.EMBED_DIM_DIST)

        # Calculate total input dimension
        input_dim = Config.EMBED_DIM_SEQ + Config.EMBED_DIM_LOOP + Config.EMBED_DIM_DIST

        # ------------------------------------------------------------------
        # 2. Stem
        # ------------------------------------------------------------------
        # Projects concatenated embeddings to the residual stream width (384)
        self.stem = nn.GRU(
            input_size=input_dim,
            hidden_size=Config.HIDDEN_DIM // 2,
            batch_first=True,
            bidirectional=True,
        )

        # ------------------------------------------------------------------
        # 3. Backbone
        # ------------------------------------------------------------------
        self.layers = nn.ModuleList(
            [
                ChannelScaledResidualBlock(Config.HIDDEN_DIM, Config.DROPOUT)
                for _ in range(Config.NUM_LAYERS)
            ]
        )

        # ------------------------------------------------------------------
        # 4. Aggregation
        # ------------------------------------------------------------------
        # Aggregates Stem + 6 Blocks = 7 sources
        self.mixture = ScalarMixture(Config.NUM_LAYERS + 1)

        # ------------------------------------------------------------------
        # 5. Output Head
        # ------------------------------------------------------------------
        self.head = nn.Linear(Config.HIDDEN_DIM, len(Config.TARGET_COLS))

    def forward(self, seq, loop, dist):
        """
        Args:
            seq (torch.Tensor): Sequence indices (Batch, Seq_Len)
            loop (torch.Tensor): Loop type indices (Batch, Seq_Len)
            dist (torch.Tensor): Structure distances (Batch, Seq_Len)

        Returns:
            torch.Tensor: Predictions (Batch, Seq_Len, 3)
        """
        # 1. Embeddings
        emb_seq = self.seq_embed(seq)  # (B, L, 128)
        emb_loop = self.loop_embed(loop)  # (B, L, 64)
        emb_dist = self.dist_embed(dist)  # (B, L, 64)

        # Concatenate inputs
        x = torch.cat([emb_seq, emb_loop, emb_dist], dim=-1)  # (B, L, 256)

        # 2. Stem
        x, _ = self.stem(x)  # (B, L, 384)

        # Collect outputs for mixture (starting with stem output)
        layer_outputs = [x]

        # 3. Backbone
        for layer in self.layers:
            x = layer(x)
            layer_outputs.append(x)

        # 4. Aggregation
        x_agg = self.mixture(layer_outputs)  # (B, L, 384)

        # 5. Head
        logits = self.head(x_agg)  # (B, L, 3)

        return logits
