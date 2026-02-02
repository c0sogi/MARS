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


class GLUResBlock(nn.Module):
    """
    Pre-Activation Direct GLU Residual Block with Batch Normalization.
    Structure: x_out = x + Dropout(GLU(Linear(BN(x))))
    Cite Lesson 49: Batch Normalization vs Layer Normalization
    Cite Lesson 51: Direct Gated topology (Projection d->2d followed by GLU reduction to d)
    Cite Lesson 54: Pre-Activation Topology
    """

    def __init__(self, in_features, dropout=0.35):
        super().__init__()
        self.norm = nn.BatchNorm1d(in_features)
        # GLU splits input in half, so we project to 2 * in_features
        self.fc = nn.Linear(in_features, in_features * 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        x = self.fc(x)
        x = F.glu(x, dim=-1)
        x = self.dropout(x)
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
                stage_blocks.append(proj)
                current_dim = stage_dim

            # Stack Residual Blocks
            for _ in range(BLOCKS_PER_STAGE):
                # Cite Lesson 84: Progressive Compression (Funneling) over Isotropic Regularized Blocks
                # Removing DropPath (Stochastic Depth) as it adds complexity without gain.
                stage_blocks.append(GLUResBlock(current_dim, dropout=DROPOUT))
                global_block_idx += 1

            self.stages.append(nn.Sequential(*stage_blocks))

        # Output Head
        self.head = nn.Linear(BACKBONE_STAGES[-1], 1)
        # Removed explicit initialization for stem/head/proj to rely on PyTorch defaults (Cite Lesson 71)

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
