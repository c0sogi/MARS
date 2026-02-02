import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from library.config import (
    EMBED_DIM,
    BACKBONE_STAGES,
    BLOCKS_PER_STAGE,
    DROP_PATH_MAX,
    DROPOUT,
)


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample.
    """

    def __init__(self, drop_prob=0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        # Work with any number of dimensions, assuming batch size is dim 0
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output


class GLUResBlock(nn.Module):
    """
    Pre-Activation Direct GLU Residual Block.
    Structure: x_out = x + DropPath(Dropout(GLU(Linear(Norm(x)))))
    Cite Lesson 00051: Simplicity in Gated Residual Networks (Direct GLU vs MLP-GLU).
    Cite Lesson 00071: Avoid blanket Xavier init for GLU (Use Defaults/Kaiming).
    """

    def __init__(self, in_features, drop_path=0.0, dropout=0.35):
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        # GLU splits input in half, so we project to 2 * in_features
        self.fc1 = nn.Linear(in_features, in_features * 2)
        self.dropout = nn.Dropout(dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        # Rely on PyTorch defaults (Kaiming Uniform) for Linear layers

    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        x = self.fc1(x)
        x = F.glu(x, dim=-1)
        x = self.dropout(x)
        x = self.drop_path(x)
        return shortcut + x


class TransformerStream(nn.Module):
    """
    Processes the categorical sequence data (f_27).
    Decomposed into 10 tokens, embedded, added with learnable pos embedding,
    and processed by a Transformer Encoder with GELU activation.
    """

    def __init__(self):
        super().__init__()
        # 26 chars + potential padding/margin, 30 is safe
        self.embed = nn.Embedding(30, EMBED_DIM)
        # Learnable Positional Embeddings: 10 tokens
        self.pos_embed = nn.Parameter(torch.zeros(1, 10, EMBED_DIM))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM,
            nhead=4,
            dim_feedforward=128,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self._init_weights()

    def _init_weights(self):
        # Initialize Positional Embeddings with Low Variance Random Noise
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)

        # Initialize Transformer params with Xavier Uniform
        for p in self.transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x_cat):
        # x_cat: (B, 10)
        x = self.embed(x_cat)  # (B, 10, 32)
        x = x + self.pos_embed
        x = self.transformer(x)
        x = x.flatten(1)  # (B, 320)
        return x


class SustainedHybridModel(nn.Module):
    """
    Stochastic-Depth Sustained-Hybrid Network.
    Fuses Transformer stream and Continuous stream, processes via a
    deep residual backbone with stochastic depth, and outputs a prediction.
    """

    def __init__(self):
        super().__init__()

        # Stream 1: Categorical
        self.transformer_stream = TransformerStream()

        # Stream 2: Continuous (30 features) - No learnable layers initially

        # Fusion Stem
        # Transformer output: 10 * 32 = 320
        # Continuous input: 30
        fusion_dim = 10 * EMBED_DIM + 30
        self.stem = nn.Linear(fusion_dim, BACKBONE_STAGES[0])

        # Backbone
        self.stages = nn.ModuleList()
        current_dim = BACKBONE_STAGES[0]

        total_blocks = len(BACKBONE_STAGES) * BLOCKS_PER_STAGE
        global_block_idx = 0

        for i, stage_dim in enumerate(BACKBONE_STAGES):
            stage_blocks = []

            # Transition/Projection if dimension changes (except first stage which matches stem)
            if i > 0:
                proj = nn.Linear(BACKBONE_STAGES[i - 1], stage_dim)
                nn.init.xavier_uniform_(proj.weight)
                nn.init.zeros_(proj.bias)
                stage_blocks.append(proj)
                current_dim = stage_dim

            # Stack Residual Blocks
            for _ in range(BLOCKS_PER_STAGE):
                # Linear schedule for DropPath
                dpr = DROP_PATH_MAX * global_block_idx / (total_blocks - 1)
                stage_blocks.append(
                    GLUResBlock(current_dim, drop_path=dpr, dropout=DROPOUT)
                )
                global_block_idx += 1

            self.stages.append(nn.Sequential(*stage_blocks))

        # Output Head
        self.head = nn.Linear(BACKBONE_STAGES[-1], 1)
        # Rely on PyTorch defaults (Kaiming Uniform) for Stem and Head
        # Cite Lesson 00071

    def forward(self, x_cat, x_cont):
        # Stream 1
        x1 = self.transformer_stream(x_cat)

        # Stream 2
        x2 = x_cont

        # Fusion
        x = torch.cat([x1, x2], dim=1)
        x = self.stem(x)

        # Backbone
        for stage in self.stages:
            x = stage(x)

        # Head
        return self.head(x)
