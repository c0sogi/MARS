import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config
from library.layers import RMSNorm, SwiGLU, RoPETransformerEncoderLayer


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
    Pre-RMSNorm Direct SwiGLU Residual Block.
    Structure: x + DropPath(Dropout(SwiGLU(Linear(RMSNorm(x)))))

    Note: SwiGLU splits the input in half. To maintain dimension 'dim',
    the inner Linear layer projects from 'dim' to '2 * dim'.
    """

    def __init__(self, dim, dropout=0.0, drop_path=0.0):
        super().__init__()
        self.norm = RMSNorm(dim)
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
    RoPE-Enhanced SwiGLU-RMS Network.

    Stream 1: Categorical Sequence -> Embedding -> RoPE Transformer -> Flatten
    Stream 2: Continuous Features -> Raw
    Fusion: Concatenation -> Linear Stem
    Backbone: Stacked SwiGLUBlocks with RMSNorm and Stochastic Depth
    Head: Linear -> Logits
    """

    def __init__(self):
        super().__init__()

        # ----------------------------------------------------------------------
        # Stream 1: Categorical Sequence (RoPE-Transformer)
        # ----------------------------------------------------------------------
        # +1 for potential unknown/padding if needed, though data is clean 0-25
        self.embedding = nn.Embedding(Config.VOCAB_SIZE + 1, Config.EMBED_DIM)

        self.transformer_layers = nn.ModuleList(
            [
                RoPETransformerEncoderLayer(
                    d_model=Config.EMBED_DIM,
                    nhead=Config.TRANSFORMER_HEADS,
                    # Feedforward dim usually 4x d_model in standard transformers
                    dim_feedforward=Config.EMBED_DIM * 4,
                    dropout=Config.TRANSFORMER_DROPOUT,
                    activation=Config.TRANSFORMER_ACTIVATION,
                    norm_first=Config.TRANSFORMER_NORM_FIRST,
                )
                for _ in range(Config.TRANSFORMER_LAYERS)
            ]
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
        # Backbone: RMSNorm-SwiGLU Funnel
        # ----------------------------------------------------------------------
        self.backbone = nn.ModuleList()

        # Calculate stochastic depth schedule
        # Total blocks = stages * blocks_per_stage
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
            # If current dim doesn't match stage dim (transition needed)
            # Or if it's the first stage, we might need to align if stem != stage[0]
            # In this design, stem maps to BACKBONE_INPUT_DIM which is usually equal to stage[0]
            if current_dim != stage_dim:
                # Pre-RMSNorm Transition
                self.backbone.append(RMSNorm(current_dim))
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
        self.head_norm = RMSNorm(current_dim)
        self.head = nn.Linear(current_dim, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """
        Custom weight initialization:
        - SwiGLU Blocks: Kaiming Uniform
        - Transformer: Xavier (Glorot)
        - Linear/Embeddings: Normal/Xavier
        - Biases: 0
        - RMSNorm weights: 1
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # He initialization for layers followed by Swish/ReLU-like activations
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            elif isinstance(m, (nn.LayerNorm, RMSNorm)):
                nn.init.ones_(m.weight)
                # RMSNorm doesn't have bias, but LayerNorm does
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Specific initialization for Transformer layers (Xavier)
        for layer in self.transformer_layers:
            for p in layer.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def forward(self, x_cat, x_cont):
        """
        Args:
            x_cat: LongTensor of shape (batch, sequence_length)
            x_cont: FloatTensor of shape (batch, num_cont_features)
        """
        # ----------------------------------------------------------------------
        # Stream 1: Categorical
        # ----------------------------------------------------------------------
        # (batch, seq_len) -> (batch, seq_len, embed_dim)
        x_seq = self.embedding(x_cat)

        # Pass through RoPE Transformer Encoder
        for layer in self.transformer_layers:
            x_seq = layer(x_seq)

        # Flatten: (batch, seq_len, embed_dim) -> (batch, seq_len * embed_dim)
        batch_size = x_seq.size(0)
        x_seq_flat = x_seq.reshape(batch_size, -1)

        # ----------------------------------------------------------------------
        # Fusion
        # ----------------------------------------------------------------------
        # Concatenate with continuous features
        x_fused = torch.cat([x_seq_flat, x_cont], dim=1)

        # Linear Stem
        x = self.stem(x_fused)

        # ----------------------------------------------------------------------
        # Backbone
        # ----------------------------------------------------------------------
        for layer in self.backbone:
            x = layer(x)

        # ----------------------------------------------------------------------
        # Head
        # ----------------------------------------------------------------------
        x = self.head_norm(x)
        out = self.head(x)

        return out


import math
