import torch
import torch.nn as nn
import math
from library.config import Config


class SinusoidalPositionalEmbedding(nn.Module):
    """
    Encodes signed scalar distances using sinusoidal functions.
    Preserves sign information via the phase properties of sin/cos.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # Precompute frequency terms
        # dim must be even for sin/cos split
        half_dim = dim // 2
        div_term = torch.exp(
            torch.arange(0, half_dim * 2, 2).float() * (-math.log(10000.0) / dim)
        )
        self.register_buffer("div_term", div_term)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Signed distances of shape (Batch, Seq_Len)
        Returns:
            torch.Tensor: Embeddings of shape (Batch, Seq_Len, dim)
        """
        # x: (B, L) -> (B, L, 1)
        # div_term: (dim/2)
        phase = x.unsqueeze(-1) * self.div_term

        # Compute sin and cos
        pe_sin = torch.sin(phase)
        pe_cos = torch.cos(phase)

        # Concatenate to get full dimension
        # Shape: (B, L, dim)
        pe = torch.cat([pe_sin, pe_cos], dim=-1)
        return pe


class WideBiGRUBlock(nn.Module):
    """
    A Residual Block using a Wide-Stream BiGRU with Pre-LayerNorm.
    Maintains the residual stream width W throughout.
    """

    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_dim)

        # Bidirectional GRU with hidden_size = hidden_dim // 2
        # Output size becomes hidden_dim, matching the residual stream
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Pre-LayerNorm
        residual = x
        out = self.ln(x)

        # BiGRU
        out, _ = self.gru(out)

        # Dropout and Residual connection
        out = self.dropout(out)
        return residual + out


class TargetSpecificAggregator(nn.Module):
    """
    Aggregates layer outputs using learned scalar weights specific to each target.
    """

    def __init__(self, n_layers, n_targets, hidden_dim):
        super().__init__()
        self.n_layers = n_layers
        self.n_targets = n_targets

        # Learnable weights: (n_targets, n_layers)
        # Initialized to 0 to start with uniform weighting (after softmax)
        self.weights = nn.Parameter(torch.zeros(n_targets, n_layers))

    def forward(self, layers_output):
        """
        Args:
            layers_output (list): List of tensors (B, L, W) from each layer.
        Returns:
            list: List of tensors (B, L, W), one per target.
        """
        # Stack layers: (B, L, n_layers, W)
        stacked = torch.stack(layers_output, dim=2)

        # Compute mixing weights: (n_targets, n_layers)
        alphas = torch.softmax(self.weights, dim=1)

        outputs = []
        for k in range(self.n_targets):
            # Get weights for target k: (n_layers)
            # Reshape for broadcasting: (1, 1, n_layers, 1)
            w = alphas[k].view(1, 1, self.n_layers, 1)

            # Weighted sum over layers
            # (B, L, n_layers, W) * (1, 1, n_layers, 1) -> sum dim 2 -> (B, L, W)
            agg = (stacked * w).sum(dim=2)
            outputs.append(agg)

        return outputs


class Net(nn.Module):
    """
    Target-Aware Layer-Aggregated Wide-Stream BiGRU.
    """

    def __init__(self):
        super().__init__()

        # 1. Input Embeddings
        self.seq_embed = nn.Embedding(Config.vocab_size, Config.seq_embed_dim)
        self.loop_embed = nn.Embedding(Config.loop_vocab_size, Config.loop_embed_dim)
        self.dist_embed = SinusoidalPositionalEmbedding(Config.distance_dim)

        # 2. Recurrent Stem
        # Projects concatenated inputs to the residual stream width (hidden_dim)
        self.stem = nn.GRU(
            input_size=Config.input_dim,
            hidden_size=Config.hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.stem_dropout = nn.Dropout(Config.dropout)

        # 3. Backbone (Wide-Stream Residual Blocks)
        self.layers = nn.ModuleList(
            [
                WideBiGRUBlock(Config.hidden_dim, Config.dropout)
                for _ in range(Config.n_layers)
            ]
        )

        # 4. Target-Specific Aggregator
        # We aggregate the Stem output + outputs from all N blocks
        total_layers = 1 + Config.n_layers
        self.aggregator = TargetSpecificAggregator(
            n_layers=total_layers,
            n_targets=Config.n_targets,
            hidden_dim=Config.hidden_dim,
        )

        # 5. Output Heads
        # Independent linear projection for each target
        self.heads = nn.ModuleList(
            [nn.Linear(Config.hidden_dim, 1) for _ in range(Config.n_targets)]
        )

    def forward(self, sequence, loop_type, distance, mask=None):
        # sequence: (B, L)
        # loop_type: (B, L)
        # distance: (B, L)

        # --- Embeddings ---
        emb_seq = self.seq_embed(sequence)  # (B, L, seq_dim)
        emb_loop = self.loop_embed(loop_type)  # (B, L, loop_dim)
        emb_dist = self.dist_embed(distance)  # (B, L, dist_dim)

        # Concatenate features
        x = torch.cat([emb_seq, emb_loop, emb_dist], dim=-1)  # (B, L, input_dim)

        # --- Stem ---
        x, _ = self.stem(x)
        x = self.stem_dropout(x)

        # Collect outputs (Stem is layer 0)
        layer_outputs = [x]

        # --- Backbone ---
        for layer in self.layers:
            x = layer(x)
            layer_outputs.append(x)

        # --- Aggregation ---
        # Returns a list of contexts, one for each target
        target_contexts = self.aggregator(layer_outputs)

        # --- Heads ---
        preds = []
        for k, head in enumerate(self.heads):
            # Project context to scalar: (B, L, W) -> (B, L, 1)
            out = head(target_contexts[k])
            preds.append(out)

        # Concatenate to (B, L, n_targets)
        final_pred = torch.cat(preds, dim=-1)

        return final_pred
