import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config
from library.utils import get_sinusoidal_encoding_table


class InputEmbedding(nn.Module):
    """
    Concatenates three feature channels:
    1. Atomic Sequence Embeddings
    2. Predicted Loop Type Embeddings
    3. Signed Sinusoidal Pairing Distance Embeddings
    """

    def __init__(self):
        super().__init__()
        # 1. Atomic Sequence
        self.seq_emb = nn.Embedding(len(Config.NUC_VOCAB), Config.EMBED_DIM)

        # 2. Predicted Loop Type
        self.loop_emb = nn.Embedding(len(Config.LOOP_VOCAB), Config.EMBED_DIM)

        # 3. Signed Sinusoidal Pairing Distance
        # We map signed distances (approx -107 to 107) to indices 0-255 using an offset.
        self.dist_offset = 128
        num_dist_embeddings = 256

        # Generate fixed sinusoidal table
        # Shape: (num_dist_embeddings, EMBED_DIM)
        sin_table = get_sinusoidal_encoding_table(num_dist_embeddings, Config.EMBED_DIM)
        self.dist_emb = nn.Embedding.from_pretrained(sin_table, freeze=True)

    def forward(self, seq, loop, dist):
        """
        Args:
            seq: (Batch, Seq_Len) LongTensor
            loop: (Batch, Seq_Len) LongTensor
            dist: (Batch, Seq_Len) LongTensor (Signed values)
        Returns:
            (Batch, Seq_Len, 3 * EMBED_DIM)
        """
        x_seq = self.seq_emb(seq)
        x_loop = self.loop_emb(loop)

        # Shift signed distances to positive indices for lookup
        dist_idx = dist + self.dist_offset
        # Clamp to ensure indices are valid (safety against unexpected lengths)
        dist_idx = torch.clamp(dist_idx, 0, 255)
        x_dist = self.dist_emb(dist_idx)

        # Concatenate features
        return torch.cat([x_seq, x_loop, x_dist], dim=-1)


class BiGRUStem(nn.Module):
    """
    Projects concatenated inputs to the residual stream width using a BiGRU.
    """

    def __init__(self, input_dim):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=Config.HIDDEN_DIM
            // 2,  # Bidirectional -> Total hidden = HIDDEN_DIM
            bidirectional=True,
            batch_first=True,
        )

    def forward(self, x):
        # x: (Batch, Seq_Len, Input_Dim)
        out, _ = self.gru(x)
        return out  # (Batch, Seq_Len, HIDDEN_DIM)


class ResidualBiGRUBlock(nn.Module):
    """
    Wide-Stream Residual Block with Pre-LayerNorm and BiGRU.
    Maintains constant width (HIDDEN_DIM) to avoid bottlenecks.
    """

    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(Config.HIDDEN_DIM)
        self.gru = nn.GRU(
            input_size=Config.HIDDEN_DIM,
            hidden_size=Config.HIDDEN_DIM // 2,
            bidirectional=True,
            batch_first=True,
        )
        self.dropout = nn.Dropout(Config.DROPOUT)

    def forward(self, x):
        # Pre-LayerNorm configuration
        residual = x
        out = self.norm(x)
        out, _ = self.gru(out)
        out = self.dropout(out)
        return residual + out


class ScalarMixture(nn.Module):
    """
    Aggregates outputs from the Stem and all Residual Blocks using a learnable weighted sum.
    Uses iterative accumulation for memory efficiency.
    """

    def __init__(self, n_layers):
        super().__init__()
        # One weight for Stem + one for each of the N layers
        self.weights = nn.Parameter(torch.zeros(n_layers + 1))

    def forward(self, tensors):
        """
        Args:
            tensors: List of (Batch, Seq_Len, Hidden_Dim) tensors.
                     Length should be n_layers + 1.
        """
        # Compute normalized weights
        norm_weights = F.softmax(self.weights, dim=0)

        # Initialize output with the first component
        out = tensors[0] * norm_weights[0]

        # Iteratively add the rest
        for i in range(1, len(tensors)):
            out = out + tensors[i] * norm_weights[i]

        return out


class OutputHead(nn.Module):
    """
    Shared Linear Projection to target channels.
    """

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(Config.HIDDEN_DIM, Config.NUM_TARGETS)

    def forward(self, x):
        return self.proj(x)


class WideResBiGRU(nn.Module):
    """
    Synchronized-Augmented Wide-Stream Residual BiGRU Model.
    """

    def __init__(self):
        super().__init__()

        self.embedding = InputEmbedding()

        # Calculate input dimension for the stem
        # 3 channels * EMBED_DIM
        stem_input_dim = 3 * Config.EMBED_DIM
        self.stem = BiGRUStem(stem_input_dim)

        # Residual Backbone
        self.blocks = nn.ModuleList(
            [ResidualBiGRUBlock() for _ in range(Config.N_LAYERS)]
        )

        # Scalar Aggregation
        self.mixture = ScalarMixture(Config.N_LAYERS)

        # Output Head
        self.head = OutputHead()

    def forward(self, sequence, loop_type, distance):
        # 1. Embed Inputs
        x = self.embedding(sequence, loop_type, distance)

        # 2. Process through Stem
        h_stem = self.stem(x)

        # 3. Process through Residual Blocks
        # Collect outputs for aggregation
        layer_outputs = [h_stem]
        current_h = h_stem

        for block in self.blocks:
            current_h = block(current_h)
            layer_outputs.append(current_h)

        # 4. Aggregate Features
        h_agg = self.mixture(layer_outputs)

        # 5. Predict Targets
        logits = self.head(h_agg)

        return logits
