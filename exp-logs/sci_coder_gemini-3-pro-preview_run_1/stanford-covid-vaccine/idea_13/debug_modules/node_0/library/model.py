import torch
import torch.nn as nn
import math
from library.config import Config


class SinusoidalDistanceEmbedding(nn.Module):
    """
    Encodes scalar distances using sinusoidal functions.
    Preserves geometric information for the neural network by projecting
    signed float distances into a high-dimensional vector space.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # Calculate the division term for the sinusoidal formulas
        # pe(pos, 2i) = sin(pos / 10000^(2i/dim))
        # pe(pos, 2i+1) = cos(pos / 10000^(2i/dim))
        # We compute the term 1 / 10000^(2i/dim) = exp(2i * -log(10000)/dim)
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * -(math.log(10000.0) / dim)
        )
        self.register_buffer("div_term", div_term)

    def forward(self, distances):
        """
        Args:
            distances: (Batch, Seq_Len) Tensor of signed float distances.
        Returns:
            (Batch, Seq_Len, Dim) Tensor of embeddings.
        """
        # Create shape [Batch, Seq_Len, 1] for broadcasting
        d = distances.unsqueeze(-1)

        # Calculate phase: pos * div_term
        # Shape: [Batch, Seq_Len, Dim/2]
        phase = d * self.div_term

        # Initialize embedding tensor
        batch_size, seq_len = distances.shape
        emb = torch.zeros(batch_size, seq_len, self.dim, device=distances.device)

        # Fill even indices with sin, odd with cos
        emb[..., 0::2] = torch.sin(phase)
        emb[..., 1::2] = torch.cos(phase)

        return emb


class KmerBiGRU(nn.Module):
    """
    K-mer Enhanced Distance-Aware BiGRU with Input Injection.

    Features:
    - 3-mer Sequence Embeddings for local stacking context.
    - Sinusoidal Distance Embeddings for geometric structure.
    - Deep BiGRU backbone with Pre-LayerNorm Residual connections.
    - Input Injection: Concatenates raw features to the input of every layer
      to preserve signal in deep networks.
    """

    def __init__(self):
        super().__init__()

        # Configuration
        self.hidden_dim = Config.HIDDEN_DIM
        self.embed_dim = Config.EMBED_DIM
        self.num_layers = Config.NUM_LAYERS
        self.use_injection = Config.USE_INPUT_INJECTION
        self.num_targets = Config.NUM_TARGETS

        # 1. Embeddings
        self.kmer_embedding = nn.Embedding(
            Config.K_MER_VOCAB_SIZE, self.embed_dim, padding_idx=0
        )
        self.loop_embedding = nn.Embedding(
            Config.LOOP_VOCAB_SIZE, self.embed_dim, padding_idx=0
        )
        self.dist_embedding = SinusoidalDistanceEmbedding(self.embed_dim)

        # Total dimension of the concatenated input features
        self.input_feature_dim = self.embed_dim * 3

        # 2. Backbone (Deep BiGRU with Input Injection)
        self.gru_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.dropout = nn.Dropout(Config.DROPOUT)

        # Layer 0: Stem Layer
        # Projects raw features to hidden space. No residual connection here.
        # Output dim is 2 * hidden_dim because it's bidirectional.
        self.gru_layers.append(
            nn.GRU(
                input_size=self.input_feature_dim,
                hidden_size=self.hidden_dim,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )
        )

        # Layers 1 to N-1: Residual Blocks with Input Injection
        for i in range(1, self.num_layers):
            # Determine input size for this layer
            # Base input is the output of previous BiGRU (2 * hidden)
            gru_input_size = self.hidden_dim * 2

            # If injection is enabled, we concatenate the original features
            if self.use_injection:
                gru_input_size += self.input_feature_dim

            self.gru_layers.append(
                nn.GRU(
                    input_size=gru_input_size,
                    hidden_size=self.hidden_dim,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=True,
                )
            )

            # LayerNorm for the Pre-LN Residual connection
            # We need one norm per residual block
            self.layer_norms.append(nn.LayerNorm(self.hidden_dim * 2))

        # Final LayerNorm after the last residual block
        self.layer_norms.append(nn.LayerNorm(self.hidden_dim * 2))

        # 3. Output Head
        self.head = nn.Linear(self.hidden_dim * 2, self.num_targets)

    def forward(self, seq_inputs, pair_dists, loop_types):
        """
        Args:
            seq_inputs: (B, L) LongTensor of K-mer indices.
            pair_dists: (B, L) FloatTensor of signed distances.
            loop_types: (B, L) LongTensor of loop type indices.
        Returns:
            logits: (B, L, Num_Targets)
        """
        # 1. Generate Embeddings
        x_kmer = self.kmer_embedding(seq_inputs)
        x_loop = self.loop_embedding(loop_types)
        x_dist = self.dist_embedding(pair_dists)

        # Concatenate to form the raw input features (B, L, 3*Embed_Dim)
        features = torch.cat([x_kmer, x_loop, x_dist], dim=-1)
        features = self.dropout(features)

        # 2. Backbone Processing

        # Layer 0 (Stem)
        # Transform raw features to hidden state space
        current_state, _ = self.gru_layers[0](features)
        current_state = self.dropout(current_state)

        # Layers 1 to N-1 (Residual Blocks)
        # Using Pre-LayerNorm configuration: x = x + F(Norm(x))
        for i in range(1, self.num_layers):
            residual = current_state

            # Norm before the sub-layer (Pre-LN)
            # layer_norms[i-1] corresponds to the input of block i
            normed_state = self.layer_norms[i - 1](current_state)

            # Prepare input for GRU
            if self.use_injection:
                # Concatenate normed hidden state with original features
                # This ensures raw signal is available at every depth
                gru_input = torch.cat([normed_state, features], dim=-1)
            else:
                gru_input = normed_state

            # GRU Pass
            out, _ = self.gru_layers[i](gru_input)
            out = self.dropout(out)

            # Residual Connection
            current_state = residual + out

        # Final Normalization
        current_state = self.layer_norms[-1](current_state)

        # 3. Prediction Head
        logits = self.head(current_state)

        return logits
