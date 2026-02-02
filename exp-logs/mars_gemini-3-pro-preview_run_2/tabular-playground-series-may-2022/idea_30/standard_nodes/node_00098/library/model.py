import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config
from library.layers import SwiGLU


class StochasticDepth(nn.Module):
    """
    Stochastic Depth (DropPath) layer.
    Randomly drops residual paths during training to regularize the network.
    """

    def __init__(self, prob: float):
        super().__init__()
        self.prob = prob

    def forward(self, x):
        if not self.training or self.prob == 0.0:
            return x

        keep_prob = 1.0 - self.prob
        # Compute shape for broadcasting: (batch_size, 1, 1, ...)
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)

        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize

        # Scale by keep_prob to maintain expected value
        return x.div(keep_prob) * random_tensor


class SwiGLUBlock(nn.Module):
    """
    Pre-LayerNorm Direct SwiGLU Residual Block.
    Structure: x + DropPath(Dropout(SwiGLU(Linear(LayerNorm(x)))))
    """

    def __init__(self, dim, dropout=0.0, drop_path=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        # Project to 2*dim because SwiGLU will slice it back to dim
        self.linear = nn.Linear(dim, 2 * dim)
        self.act = SwiGLU()
        self.dropout = nn.Dropout(dropout)
        self.drop_path = (
            StochasticDepth(drop_path) if drop_path > 0.0 else nn.Identity()
        )

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.linear(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.drop_path(x)
        return residual + x


class RoPESwiGLURMSNet(nn.Module):
    """
    Hybrid SwiGLU Network with Absolute Positional Embeddings and LayerNorm.

    Stream 1: Categorical Sequence -> Embedding + PosEmb -> Transformer -> Flatten
    Stream 2: Continuous Features -> Raw
    Fusion: Concatenation -> Linear Stem
    Backbone: Stacked SwiGLUBlocks with LayerNorm and Stochastic Depth
    Head: Linear -> Logits
    """

    def __init__(self):
        super().__init__()

        # ----------------------------------------------------------------------
        # Stream 1: Categorical Sequence (Transformer)
        # ----------------------------------------------------------------------
        self.embedding = nn.Embedding(Config.VOCAB_SIZE + 1, Config.EMBED_DIM)

        # Absolute Positional Embeddings (Cite solution_lesson_node_00096)
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, Config.SEQUENCE_LENGTH, Config.EMBED_DIM)
        )

        # Standard Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=Config.EMBED_DIM,
            nhead=Config.TRANSFORMER_HEADS,
            dim_feedforward=Config.EMBED_DIM * 4,
            dropout=Config.TRANSFORMER_DROPOUT,
            activation=Config.TRANSFORMER_ACTIVATION,  # Explicit GELU (Cite solution_lesson_node_00068)
            batch_first=True,
            norm_first=Config.TRANSFORMER_NORM_FIRST,  # Post-Norm (Cite solution_lesson_node_00095)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=Config.TRANSFORMER_LAYERS
        )

        # ----------------------------------------------------------------------
        # Fusion Layer
        # ----------------------------------------------------------------------
        # Flattened transformer output size
        transformer_out_dim = Config.SEQUENCE_LENGTH * Config.EMBED_DIM

        # Total input to stem
        fusion_input_dim = transformer_out_dim + Config.NUM_CONT_FEATURES

        # Linear Stem
        self.stem = nn.Linear(fusion_input_dim, Config.BACKBONE_INPUT_DIM)

        # ----------------------------------------------------------------------
        # Backbone: LayerNorm-SwiGLU Funnel
        # ----------------------------------------------------------------------
        self.backbone = nn.ModuleList()

        # Calculate stochastic depth schedule
        total_blocks = len(Config.BACKBONE_STAGES) * Config.BLOCKS_PER_STAGE
        dpr = [
            x.item()
            for x in torch.linspace(
                Config.STOCHASTIC_DEPTH_MIN, Config.STOCHASTIC_DEPTH_MAX, total_blocks
            )
        ]

        current_dim = Config.BACKBONE_INPUT_DIM
        block_idx = 0

        for i, stage_dim in enumerate(Config.BACKBONE_STAGES):
            # Transition
            if current_dim != stage_dim:
                # Pre-Norm Transition (Cite solution_lesson_node_00088)
                self.backbone.append(nn.LayerNorm(current_dim))
                self.backbone.append(nn.Linear(current_dim, stage_dim))
                current_dim = stage_dim

            # Stack Blocks
            for _ in range(Config.BLOCKS_PER_STAGE):
                self.backbone.append(
                    SwiGLUBlock(
                        dim=current_dim,
                        dropout=Config.BACKBONE_DROPOUT,
                        drop_path=dpr[block_idx],
                    )
                )
                block_idx += 1

        # ----------------------------------------------------------------------
        # Output Head
        # ----------------------------------------------------------------------
        self.head_norm = nn.LayerNorm(current_dim)
        self.head = nn.Linear(current_dim, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """
        Custom weight initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # He initialization for SwiGLU/ReLU
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Initialize Positional Embeddings with low variance (Cite solution_lesson_node_00081)
        nn.init.normal_(self.pos_embedding, mean=0.0, std=0.02)

        # Transformer initialization is handled by PyTorch defaults (Xavier) or can be explicit
        # PyTorch defaults are generally fine, but we can enforce Xavier for attention
        for p in self.transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x_cat, x_cont):
        # Stream 1: Categorical
        x_seq = self.embedding(x_cat)
        # Add positional embeddings
        x_seq = x_seq + self.pos_embedding

        x_seq = self.transformer(x_seq)

        batch_size = x_seq.size(0)
        x_seq_flat = x_seq.reshape(batch_size, -1)

        # Fusion
        x_fused = torch.cat([x_seq_flat, x_cont], dim=1)
        x = self.stem(x_fused)

        # Backbone
        for layer in self.backbone:
            x = layer(x)

        # Head
        x = self.head_norm(x)
        out = self.head(x)

        return out


import math
