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


class StructureInjectedBlock(nn.Module):
    """
    A Residual Block that re-injects static structural context into the processing stream.

    Architecture:
    x_norm = LayerNorm(x)
    h_in = Concat(x_norm, structural_context)
    h_rnn = BiLSTM(h_in)
    out = x + Dropout(h_rnn)
    """

    def __init__(self, stream_dim, struct_dim, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(stream_dim)

        # Input to LSTM is the residual stream width + structural embedding width
        # We project back to stream_dim (bidirectional: hidden_size = stream_dim // 2)
        self.lstm = nn.LSTM(
            input_size=stream_dim + struct_dim,
            hidden_size=stream_dim // 2,
            batch_first=True,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, struct_emb):
        """
        Args:
            x: Residual stream tensor (N, L, stream_dim)
            struct_emb: Static structural context tensor (N, L, struct_dim)
        """
        # Pre-LayerNorm configuration
        norm_x = self.norm(x)

        # Inject Structure: Concatenate static structure to dynamic stream
        h_in = torch.cat([norm_x, struct_emb], dim=-1)

        # BiLSTM Processing
        h_rnn, _ = self.lstm(h_in)

        # Residual Connection
        return x + self.dropout(h_rnn)


class StructureInjectedWideResBiLSTM(nn.Module):
    """
    Structure-Injected Wide-Stream Residual BiLSTM.

    Features:
    - Heterogeneous Embeddings (Seq, Loop, Dist)
    - Wide Residual Stream (512 dim)
    - Structure Injection at every block
    - Scalar Mixture Aggregation
    """

    def __init__(self):
        super().__init__()

        # --- 1. Embeddings ---
        self.seq_embed = nn.Embedding(Config.SEQ_VOCAB_SIZE, Config.SEQ_EMBED_DIM)
        self.loop_embed = nn.Embedding(Config.LOOP_VOCAB_SIZE, Config.LOOP_EMBED_DIM)
        # Distance embedding is functional (sinusoidal), so no learnable embedding layer needed.

        # Dimensions
        self.struct_dim = Config.LOOP_EMBED_DIM + Config.DIST_EMBED_DIM
        self.input_dim = Config.SEQ_EMBED_DIM + self.struct_dim
        self.stream_dim = Config.HIDDEN_SIZE  # 512

        # --- 2. High-Fidelity Recurrent Stem ---
        # Projects full input (Seq + Structure) to the residual stream width
        self.stem = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.stream_dim // 2,
            batch_first=True,
            bidirectional=True,
        )
        # Note: No dropout applied to stem output

        # --- 3. Backbone: Structure-Injected Residual Blocks ---
        self.blocks = nn.ModuleList(
            [
                StructureInjectedBlock(
                    stream_dim=self.stream_dim,
                    struct_dim=self.struct_dim,
                    dropout=Config.DROPOUT,
                )
                for _ in range(Config.NUM_LAYERS)
            ]
        )

        # --- 4. Aggregation: Scalar Mixture ---
        # Aggregates outputs from Stem + 6 Blocks (Total 7 layers)
        self.mixture = ScalarMixture(Config.NUM_LAYERS + 1)

        # --- 5. Output Head ---
        self.head = nn.Linear(self.stream_dim, Config.NUM_CLASSES)

    def forward(self, seq, loop, dist):
        """
        Args:
            seq: Sequence indices (N, L)
            loop: Loop type indices (N, L)
            dist: Signed distance values (N, L)
        """
        # 1. Generate Embeddings
        e_seq = self.seq_embed(seq)  # (N, L, 128)
        e_loop = self.loop_embed(loop)  # (N, L, 64)
        e_dist = get_sinusoidal_encoding(dist, Config.DIST_EMBED_DIM)  # (N, L, 64)

        # 2. Construct Structural Context Vector
        # This vector represents the static topology and is used in Stem and all Blocks
        e_struct = torch.cat([e_loop, e_dist], dim=-1)  # (N, L, 128)

        # 3. Stem Execution
        # Input: Sequence + Structure
        x_in = torch.cat([e_seq, e_struct], dim=-1)  # (N, L, 256)
        x_stem, _ = self.stem(x_in)  # (N, L, 512)

        # 4. Backbone Execution
        # Collect outputs for scalar mixture
        layer_outputs = [x_stem]
        x = x_stem

        for block in self.blocks:
            # Pass structural context to every block
            x = block(x, e_struct)
            layer_outputs.append(x)

        # 5. Aggregation
        # Combine Stem and Block outputs
        x_agg = self.mixture(layer_outputs)  # (N, L, 512)

        # 6. Prediction
        logits = self.head(x_agg)  # (N, L, 3)

        return logits
