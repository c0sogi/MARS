import torch
import torch.nn as nn
from library.config import Config
from library.layers import SinusoidalPairingEmbedding, ZoneoutBiGRUBlock, ScalarMixture


class ZoneoutWideResBiGRU(nn.Module):
    """
    Zoneout-Regularized Wide-Stream Residual BiGRU Model.

    Architecture:
    1. Multi-Modal Embeddings (Sequence, Loop Type, Pairing Distance)
    2. Recurrent Stem (BiGRU Projection)
    3. Stack of Zoneout-Regularized Residual BiGRU Blocks (Pre-LN)
    4. Scalar Mixture Aggregation
    5. Linear Output Head
    """

    def __init__(self):
        super().__init__()

        # =====================================================================
        # 1. Embeddings
        # =====================================================================
        self.embed_dim = Config.EMBED_DIM

        # Atomic Sequence Embedding (A, G, C, U)
        self.seq_embed = nn.Embedding(Config.VOCAB_SIZE, self.embed_dim)

        # Predicted Loop Type Embedding (S, M, I, B, H, E, X)
        self.loop_embed = nn.Embedding(Config.LOOP_TYPES, self.embed_dim)

        # Signed Sinusoidal Pairing Distance Embedding
        self.pair_embed = SinusoidalPairingEmbedding(self.embed_dim)

        # Total input dimension to the stem
        # 3 channels * 128 dim = 384
        self.input_dim = self.embed_dim * 3

        # =====================================================================
        # 2. Recurrent Stem
        # =====================================================================
        self.hidden_dim = Config.HIDDEN_DIM  # 512

        # Projects inputs to the residual stream width W=512
        # BiGRU: hidden_size = 512 // 2 = 256 per direction
        self.stem = nn.GRU(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim // 2,
            batch_first=True,
            bidirectional=True,
        )

        # =====================================================================
        # 3. Backbone: Zoneout-Regularized Residual Blocks
        # =====================================================================
        self.num_layers = Config.NUM_LAYERS
        self.blocks = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(self.num_layers):
            # Pre-LayerNorm
            self.norms.append(nn.LayerNorm(self.hidden_dim))

            # Zoneout BiGRU Block
            # Input size matches hidden_dim (residual stream width)
            # Hidden size is half of hidden_dim (for bidirectional concat)
            self.blocks.append(
                ZoneoutBiGRUBlock(
                    input_size=self.hidden_dim,
                    hidden_size=self.hidden_dim // 2,
                    zoneout_prob=Config.ZONEOUT_PROB,
                )
            )

        # =====================================================================
        # 4. Aggregation
        # =====================================================================
        # Aggregates outputs from Stem + 6 Blocks
        self.mixture = ScalarMixture(num_layers=self.num_layers + 1)

        # =====================================================================
        # 5. Output Head
        # =====================================================================
        self.head = nn.Linear(self.hidden_dim, Config.NUM_TARGETS)

    def forward(self, sequence, loop_type, pair_dist):
        """
        Args:
            sequence (torch.Tensor): (Batch, Seq_Len) LongTensor
            loop_type (torch.Tensor): (Batch, Seq_Len) LongTensor
            pair_dist (torch.Tensor): (Batch, Seq_Len) FloatTensor

        Returns:
            torch.Tensor: Predictions (Batch, Seq_Len, Num_Targets)
        """
        # ---------------------------------------------------------------------
        # 1. Embeddings
        # ---------------------------------------------------------------------
        # (B, L) -> (B, L, D)
        emb_seq = self.seq_embed(sequence)
        emb_loop = self.loop_embed(loop_type)
        emb_pair = self.pair_embed(pair_dist)

        # Concatenate: (B, L, 3*D)
        x = torch.cat([emb_seq, emb_loop, emb_pair], dim=-1)

        # ---------------------------------------------------------------------
        # 2. Recurrent Stem
        # ---------------------------------------------------------------------
        # (B, L, 3*D) -> (B, L, W)
        x, _ = self.stem(x)

        # We collect outputs for the mixture
        layer_outputs = [x]

        # ---------------------------------------------------------------------
        # 3. Residual Backbone
        # ---------------------------------------------------------------------
        # current_features acts as the residual stream
        current_features = x

        for norm_layer, block in zip(self.norms, self.blocks):
            # Pre-LayerNorm: Norm -> Block -> Add
            # 1. Norm
            normed_features = norm_layer(current_features)

            # 2. Block (Zoneout BiGRU)
            block_out = block(normed_features)

            # 3. Residual Connection
            current_features = current_features + block_out

            # Store for aggregation
            layer_outputs.append(current_features)

        # ---------------------------------------------------------------------
        # 4. Aggregation
        # ---------------------------------------------------------------------
        # Weighted sum of [Stem, Block1, ..., Block6]
        # (B, L, W)
        aggregated_features = self.mixture(layer_outputs)

        # ---------------------------------------------------------------------
        # 5. Output Head
        # ---------------------------------------------------------------------
        # (B, L, Num_Targets)
        logits = self.head(aggregated_features)

        return logits
