import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class ScalarMixture(nn.Module):
    """
    Computes a learnable weighted sum of a list of tensors.
    Used to aggregate representations from different layers of the model.
    """

    def __init__(self, num_layers):
        super().__init__()
        self.num_layers = num_layers
        # Learnable weights for each layer, initialized to 0 (effectively uniform after softmax)
        self.weights = nn.Parameter(torch.zeros(num_layers))
        # Learnable scaling factor
        self.gamma = nn.Parameter(torch.ones(1))

    def forward(self, tensors):
        """
        Args:
            tensors: List of tensors, each of shape (Batch, Seq, Dim)
        Returns:
            Tensor of shape (Batch, Seq, Dim)
        """
        # Stack tensors: (Batch, Seq, Dim, Num_Layers)
        stacked = torch.stack(tensors, dim=-1)

        # Compute normalized weights
        norm_weights = F.softmax(self.weights, dim=0)

        # Weighted sum across the layer dimension
        weighted_sum = torch.sum(stacked * norm_weights, dim=-1)

        return self.gamma * weighted_sum


class ResidualBiGRUBlock(nn.Module):
    """
    A Wide-Stream Residual Block with Pre-LayerNorm configuration.
    Structure: Input -> LayerNorm -> BiGRU -> Dropout -> Residual Add
    """

    def __init__(self, hidden_dim, dropout):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_dim)
        # Wide stream: BiGRU hidden size sums to hidden_dim (e.g., 192*2 = 384)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x

        # Pre-LayerNorm
        out = self.ln(x)

        # BiGRU
        out, _ = self.gru(out)

        # Dropout
        out = self.dropout(out)

        # Residual Connection
        return residual + out


class RNAModel(nn.Module):
    """
    Discrete-Topology Wide-Stream Residual BiGRU Model.
    """

    def __init__(self, config=Config):
        super().__init__()

        # 1. Heterogeneous Feature Embeddings
        self.seq_embed = nn.Embedding(config.VOCAB_SIZE_SEQ, config.EMBED_DIM_SEQ)
        self.loop_embed = nn.Embedding(config.VOCAB_SIZE_LOOP, config.EMBED_DIM_LOOP)
        self.dist_embed = nn.Embedding(config.VOCAB_SIZE_DIST, config.EMBED_DIM_DIST)

        # 2. Recurrent Stem
        # Projects concatenated inputs to the residual stream width
        self.stem_gru = nn.GRU(
            input_size=config.INPUT_DIM,
            hidden_size=config.HIDDEN_DIM // 2,
            batch_first=True,
            bidirectional=True,
        )
        # Note: No dropout after stem as per specification

        # 3. Backbone: Stack of Residual Blocks
        self.blocks = nn.ModuleList(
            [
                ResidualBiGRUBlock(config.HIDDEN_DIM, config.DROPOUT)
                for _ in range(config.NUM_LAYERS)
            ]
        )

        # 4. Aggregation: Scalar Mixture
        # Aggregates outputs from the Stem + all Blocks
        self.scalar_mixture = ScalarMixture(num_layers=config.NUM_LAYERS + 1)

        # 5. Output Head
        self.head = nn.Linear(config.HIDDEN_DIM, config.NUM_TARGETS)

    def forward(self, sequence, loop_type, structure_dist):
        """
        Args:
            sequence: (Batch, Seq) LongTensor
            loop_type: (Batch, Seq) LongTensor
            structure_dist: (Batch, Seq) LongTensor
        Returns:
            logits: (Batch, Seq, Num_Targets) FloatTensor
        """
        # Embed Inputs
        emb_seq = self.seq_embed(sequence)  # (B, L, 128)
        emb_loop = self.loop_embed(loop_type)  # (B, L, 64)
        emb_dist = self.dist_embed(structure_dist)  # (B, L, 64)

        # Early Fusion
        x = torch.cat([emb_seq, emb_loop, emb_dist], dim=-1)  # (B, L, 256)

        # Stem Processing
        x, _ = self.stem_gru(x)  # (B, L, 384)

        # Collect layer outputs for aggregation
        layer_outputs = [x]

        # Pass through Residual Blocks
        for block in self.blocks:
            x = block(x)
            layer_outputs.append(x)

        # Aggregate representations
        x_aggregated = self.scalar_mixture(layer_outputs)

        # Prediction
        logits = self.head(x_aggregated)  # (B, L, 3)

        return logits
