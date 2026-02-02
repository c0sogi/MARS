import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from library.config import Config


class SinusoidalDistanceEncoding(nn.Module):
    """
    Computes sinusoidal encodings for signed integer distances.
    Based on the standard Transformer positional encoding but adapted for signed values.
    """

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        # Ensure d_model is even for splitting into sin/cos
        if d_model % 2 != 0:
            raise ValueError("d_model for sinusoidal encoding must be even.")

        # Precompute the division term for the sinusoidal formulas
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        self.register_buffer("div_term", div_term)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Signed integer distances. Shape [Batch, SeqLen].
        Returns:
            torch.Tensor: Sinusoidal encodings. Shape [Batch, SeqLen, d_model].
        """
        # Expand dims for broadcasting: [Batch, SeqLen, 1]
        x_expanded = x.unsqueeze(-1).float()

        # Calculate phase: [Batch, SeqLen, d_model/2]
        phase = x_expanded * self.div_term

        # Concatenate sin and cos along the last dimension -> [Batch, SeqLen, d_model]
        return torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)


class ScalarMixture(nn.Module):
    """
    Computes a learnable weighted sum of a list of tensors.
    Used to aggregate outputs from different layers of the model.
    """

    def __init__(self, n_tensors):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(n_tensors))

    def forward(self, tensor_list):
        """
        Args:
            tensor_list (list of torch.Tensor): List of tensors, each shape [B, S, D].
        Returns:
            torch.Tensor: Weighted sum, shape [B, S, D].
        """
        # Stack tensors: [B, S, D, N]
        stacked = torch.stack(tensor_list, dim=-1)

        # Normalize weights using softmax
        norm_weights = F.softmax(self.weights, dim=0)

        # Weighted sum along the last dimension
        return torch.sum(stacked * norm_weights, dim=-1)


class WideResBiGRU(nn.Module):
    """
    Wide-Stream Residual BiGRU.

    Architecture Strategy:
    1. Proportional Embeddings (Seq, Loop, Dist).
    2. Direct concatenation of embeddings (No independent normalization, Cite {solution_lesson_node_00102}).
    3. BiGRU Stem projecting to a wide hidden dimension (384).
    4. 6x Wide-Stream Residual BiGRU Blocks (Pre-LN, Dropout) maintaining full width.
    5. Scalar Mixture Aggregation to combine multi-level features.
    6. Output Head for 3 target channels.
    """

    def __init__(self):
        super().__init__()

        # --- 1. Embeddings ---
        # Sequence: 4 tokens (A, G, C, U)
        self.seq_embed = nn.Embedding(4, Config.EMBED_DIM_SEQ)

        # Loop Type: 7 tokens (S, M, I, B, H, E, X)
        self.loop_embed = nn.Embedding(7, Config.EMBED_DIM_LOOP)

        # Distance: Fixed Sinusoidal Encoding
        self.dist_embed = SinusoidalDistanceEncoding(Config.EMBED_DIM_DIST)

        # Calculate fused input dimension
        stem_input_dim = (
            Config.EMBED_DIM_SEQ + Config.EMBED_DIM_LOOP + Config.EMBED_DIM_DIST
        )

        # --- 3. Recurrent Stem ---
        # Projects concatenated inputs to the residual stream width
        # Bidirectional GRU: hidden_size = HIDDEN_DIM // 2
        self.stem = nn.GRU(
            input_size=stem_input_dim,
            hidden_size=Config.HIDDEN_DIM // 2,
            batch_first=True,
            bidirectional=True,
        )

        # --- 4. Backbone: Wide-Stream Residual Blocks ---
        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for _ in range(Config.N_LAYERS):
            # Pre-LayerNorm configuration
            self.layer_norms.append(nn.LayerNorm(Config.HIDDEN_DIM))

            # BiGRU Layer keeping the stream wide
            self.layers.append(
                nn.GRU(
                    input_size=Config.HIDDEN_DIM,
                    hidden_size=Config.HIDDEN_DIM // 2,
                    batch_first=True,
                    bidirectional=True,
                )
            )

            # Inter-layer Dropout
            self.dropouts.append(nn.Dropout(Config.DROPOUT))

        # --- 5. Aggregation ---
        # Aggregates Stem output + N_LAYERS block outputs
        self.mixture = ScalarMixture(Config.N_LAYERS + 1)

        # --- 6. Output Head ---
        self.head = nn.Linear(Config.HIDDEN_DIM, Config.NUM_CLASSES)

    def forward(self, sequence, loop_type, pair_dist):
        """
        Args:
            sequence (torch.Tensor): [Batch, SeqLen]
            loop_type (torch.Tensor): [Batch, SeqLen]
            pair_dist (torch.Tensor): [Batch, SeqLen]
        Returns:
            torch.Tensor: [Batch, SeqLen, NumClasses]
        """

        # 1. Embed
        e_seq = self.seq_embed(sequence)
        e_loop = self.loop_embed(loop_type)
        e_dist = self.dist_embed(pair_dist)

        # 2. Concatenate (No Norm, Cite {solution_lesson_node_00102})
        x = torch.cat([e_seq, e_loop, e_dist], dim=-1)

        # 3. Stem
        stem_out, _ = self.stem(x)

        # Initialize list for mixture with stem output
        layer_outputs = [stem_out]
        curr = stem_out

        # 5. Backbone
        for i in range(Config.N_LAYERS):
            residual = curr

            # Pre-LN
            out = self.layer_norms[i](curr)

            # BiGRU
            out, _ = self.layers[i](out)

            # Dropout
            out = self.dropouts[i](out)

            # Residual Connection
            curr = out + residual

            layer_outputs.append(curr)

        # 6. Aggregation
        mixed_context = self.mixture(layer_outputs)  # [B, S, 512]

        # 7. Head
        logits = self.head(mixed_context)  # [B, S, 3]

        return logits
