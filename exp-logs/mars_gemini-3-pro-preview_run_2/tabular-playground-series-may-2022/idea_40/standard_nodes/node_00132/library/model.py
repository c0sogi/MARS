import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from library.config import Config
from library.utils import custom_weight_init


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample.
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class DirectSwiGLU(nn.Module):
    """
    Direct SwiGLU Block: Linear(d->2d) -> SwishGate -> Output(d)
    """

    def __init__(self, in_features):
        super().__init__()
        self.linear = nn.Linear(in_features, 2 * in_features)

    def forward(self, x):
        # Project d -> 2d
        h = self.linear(x)
        # Split into gate and value
        x1, x2 = h.chunk(2, dim=-1)
        # SwishGate: Swish(x1) * x2
        return F.silu(x1) * x2


class ModifiedTransformerEncoderLayer(nn.Module):
    """
    Modified Transformer Encoder Layer with Direct SwiGLU and Post-Norm.
    Structure:
      y = x + Dropout(Attn(x))
      x = Norm(y)
      z = x + Dropout(SwiGLU(x))
      out = Norm(z)
    """

    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.swiglu = DirectSwiGLU(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # 1. Self-Attention
        src2 = self.self_attn(
            src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)  # Post-Norm

        # 2. Direct SwiGLU
        src2 = self.swiglu(src)
        src = src + self.dropout2(src2)
        src = self.norm2(src)  # Post-Norm
        return src


class ResidualSwiGLUBlock(nn.Module):
    """
    Pre-LayerNorm Direct SwiGLU Residual Block.
    x_out = x + DropPath(Dropout(SwiGLU(Linear(LayerNorm(x)))))
    """

    def __init__(self, dim, drop_path=0.0, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.swiglu = DirectSwiGLU(dim)
        self.dropout = nn.Dropout(dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input_x = x
        x = self.norm(x)
        x = self.swiglu(x)
        x = self.dropout(x)
        x = self.drop_path(x)
        return input_x + x


class HybridNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        # ----------------------------------------------------------------------
        # Stream 1: Categorical Sequence
        # ----------------------------------------------------------------------
        self.embed_dim = Config.EMBED_DIM
        self.seq_len = Config.SEQ_LEN
        self.vocab_size = Config.VOCAB_SIZE

        # Token Embedding
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)

        # Positional Embedding (Learnable)
        self.pos_embed = nn.Parameter(torch.empty(1, self.seq_len, self.embed_dim))

        # Transformer Encoder
        encoder_layers = []
        for _ in range(Config.ENCODER_LAYERS):
            encoder_layers.append(
                ModifiedTransformerEncoderLayer(
                    d_model=self.embed_dim,
                    nhead=Config.ENCODER_HEADS,
                    dropout=Config.ENCODER_DROPOUT,
                )
            )
        self.encoder = nn.Sequential(*encoder_layers)

        # Affine-Free Alignment
        # Flattens (N, L, D) -> (N, L*D)
        self.flatten_dim = self.seq_len * self.embed_dim
        self.alignment_norm = nn.BatchNorm1d(self.flatten_dim, affine=False)

        # ----------------------------------------------------------------------
        # Stream 2: Continuous Features
        # ----------------------------------------------------------------------
        self.num_cont = Config.NUM_CONT_FEATURES

        # ----------------------------------------------------------------------
        # Fusion & Backbone
        # ----------------------------------------------------------------------
        fusion_input_dim = self.flatten_dim + self.num_cont

        # Linear Stem
        self.stem = nn.Linear(fusion_input_dim, Config.BACKBONE_STAGES[0])

        # Backbone Construction
        stages = []
        in_dim = Config.BACKBONE_STAGES[0]

        # Stochastic Depth Schedule
        total_blocks = sum([Config.BLOCKS_PER_STAGE] * len(Config.BACKBONE_STAGES))
        dpr = [
            x.item()
            for x in torch.linspace(0, Config.STOCHASTIC_DEPTH_MAX, total_blocks)
        ]
        block_idx = 0

        for stage_idx, out_dim in enumerate(Config.BACKBONE_STAGES):
            # Transition (Pre-Norm)
            if stage_idx > 0:
                stages.append(nn.LayerNorm(in_dim))
                stages.append(nn.Linear(in_dim, out_dim))
                in_dim = out_dim

            # Residual Blocks
            for _ in range(Config.BLOCKS_PER_STAGE):
                stages.append(
                    ResidualSwiGLUBlock(
                        dim=in_dim,
                        drop_path=dpr[block_idx],
                        dropout=Config.BACKBONE_DROPOUT,
                    )
                )
                block_idx += 1

        self.backbone = nn.Sequential(*stages)

        # ----------------------------------------------------------------------
        # Output Head
        # ----------------------------------------------------------------------
        final_dim = Config.BACKBONE_STAGES[-1]
        self.head = nn.Linear(final_dim, 1)

        # ----------------------------------------------------------------------
        # Initialization
        # ----------------------------------------------------------------------
        # 1. Apply global custom init (handles SwiGLU Kaiming, Attn Xavier, Embed Unit Var)
        self.apply(custom_weight_init)

        # 2. Explicitly initialize Positional Embeddings with Low Variance Noise
        nn.init.normal_(self.pos_embed, mean=0.0, std=Config.POS_EMBED_STD)

        # 3. Explicitly initialize Stem Bias to Uniform (override Zero init from custom_weight_init)
        # Standard uniform bound for Linear: 1 / sqrt(fan_in)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.stem.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.stem.bias, -bound, bound)

    def forward(self, x_cat, x_cont):
        """
        Args:
            x_cat: LongTensor (N, SEQ_LEN)
            x_cont: FloatTensor (N, NUM_CONT_FEATURES)
        """
        # --- Stream 1: Sequence ---
        x = self.embedding(x_cat)  # (N, L, D)
        x = x + self.pos_embed  # Add Positional Embedding
        x = self.encoder(x)  # (N, L, D)

        # Flatten and Align
        x = x.reshape(x.size(0), -1)  # (N, L*D)
        x = self.alignment_norm(x)  # Force global stats, no affine

        # --- Fusion ---
        # Concatenate with raw continuous features
        x = torch.cat([x, x_cont], dim=1)

        # Stem Projection
        x = self.stem(x)

        # --- Backbone ---
        x = self.backbone(x)

        # --- Head ---
        logits = self.head(x)
        return torch.sigmoid(logits).squeeze(-1)
