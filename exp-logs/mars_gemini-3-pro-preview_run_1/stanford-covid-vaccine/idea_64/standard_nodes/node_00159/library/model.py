import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config
from library.features import get_sinusoidal_encoding


class ScalarMixture(nn.Module):
    """
    Computes a learnable weighted sum of layer outputs.
    Formula: y = gamma * sum(softmax(w_i) * x_i)
    """

    def __init__(self, num_layers):
        super().__init__()
        # Initialize weights to 0 so softmax yields uniform distribution initially
        self.weights = nn.Parameter(torch.zeros(num_layers))
        self.gamma = nn.Parameter(torch.tensor(1.0))

    def forward(self, tensors):
        """
        Args:
            tensors: List of tensors, each shape (N, L, D)
        Returns:
            Weighted sum tensor of shape (N, L, D)
        """
        # Stack tensors: (NumLayers, N, L, D)
        stacked = torch.stack(tensors, dim=0)

        # Compute normalized weights via Softmax
        probs = F.softmax(self.weights, dim=0)  # (NumLayers)

        # Reshape for broadcasting: (NumLayers, 1, 1, 1)
        probs = probs.view(-1, 1, 1, 1)

        # Compute weighted sum
        weighted_sum = torch.sum(stacked * probs, dim=0)

        # Scale by gamma
        return self.gamma * weighted_sum


class ResidualBiGRUBlock(nn.Module):
    """
    Standard Pre-LayerNorm Residual BiGRU Block.
    Cite solution_lesson_node_00154 (Avoid re-injection)
    Cite solution_lesson_node_00135 (Pre-LayerNorm)
    Cite solution_lesson_node_00108 (GRU vs LSTM)
    """

    def __init__(self, hidden_size, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size // 2,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Pre-LayerNorm
        out = self.norm(x)
        # BiGRU
        out, _ = self.gru(out)
        # Residual
        return x + self.dropout(out)


class StabilizedWideResBiGRU(nn.Module):
    """
    Stabilized Wide-Stream Residual BiGRU.

    Features:
    - BiGRU Backbone (Cite solution_lesson_node_00108)
    - No Structure Injection in blocks (Cite solution_lesson_node_00154)
    - Stem Fidelity (No dropout after stem) (Cite solution_lesson_node_00109)
    """

    def __init__(self):
        super().__init__()

        # --- 1. Embeddings ---
        self.seq_embed = nn.Embedding(Config.SEQ_VOCAB_SIZE, Config.SEQ_EMBED_DIM)
        self.loop_embed = nn.Embedding(Config.LOOP_VOCAB_SIZE, Config.LOOP_EMBED_DIM)

        # Dimensions
        self.struct_dim = Config.LOOP_EMBED_DIM + Config.DIST_EMBED_DIM
        self.input_dim = Config.SEQ_EMBED_DIM + self.struct_dim
        self.hidden_size = Config.HIDDEN_SIZE

        # --- 2. Stem ---
        # Projects full input (Seq + Structure) to the residual stream width
        self.stem = nn.GRU(
            input_size=self.input_dim,
            hidden_size=self.hidden_size // 2,
            batch_first=True,
            bidirectional=True,
        )
        # Note: No dropout applied to stem output (Cite solution_lesson_node_00109)

        # --- 3. Backbone ---
        self.blocks = nn.ModuleList(
            [
                ResidualBiGRUBlock(
                    hidden_size=self.hidden_size,
                    dropout=Config.DROPOUT,
                )
                for _ in range(Config.NUM_LAYERS)
            ]
        )

        # --- 4. Aggregation ---
        self.mixture = ScalarMixture(Config.NUM_LAYERS + 1)

        # --- 5. Output Head ---
        self.head = nn.Linear(self.hidden_size, Config.NUM_CLASSES)

    def forward(self, seq, loop, dist):
        # 1. Generate Embeddings
        e_seq = self.seq_embed(seq)
        e_loop = self.loop_embed(loop)
        e_dist = get_sinusoidal_encoding(dist, Config.DIST_EMBED_DIM)

        # 2. Construct Input Vector
        # Concatenate all features (Early Fusion)
        x_in = torch.cat([e_seq, e_loop, e_dist], dim=-1)

        # 3. Stem Execution
        x_stem, _ = self.stem(x_in)

        # 4. Backbone Execution
        layer_outputs = [x_stem]
        x = x_stem

        for block in self.blocks:
            x = block(x)
            layer_outputs.append(x)

        # 5. Aggregation
        x_agg = self.mixture(layer_outputs)

        # 6. Prediction
        logits = self.head(x_agg)

        return logits
