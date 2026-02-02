import torch
import torch.nn as nn
import math
from library.config import Config


class ResFunnelBlock(nn.Module):
    """
    Pre-Activation Direct GLU Residual Block.
    Topology: x_out = Projection(x) + Dropout(GLU(Linear(BatchNorm(x))))

    Handles both identity residuals (when in_dim == out_dim) and
    projected residuals (when in_dim != out_dim) for downsampling/transitions.
    """

    def __init__(self, in_dim, out_dim, dropout_rate):
        super().__init__()
        self.bn = nn.BatchNorm1d(in_dim)
        # GLU halves the dimension, so Linear must output 2 * out_dim
        self.linear = nn.Linear(in_dim, out_dim * 2)
        self.glu = nn.GLU(dim=1)
        self.dropout = nn.Dropout(dropout_rate)

        # Projected Residual Connection for dimension changes
        if in_dim != out_dim:
            self.projection = nn.Linear(in_dim, out_dim)
        else:
            self.projection = nn.Identity()

    def forward(self, x):
        # Residual path (Identity or Projection)
        shortcut = self.projection(x)

        # Main path: Pre-Activation -> Linear -> GLU -> Dropout
        out = self.bn(x)
        out = self.linear(out)
        out = self.glu(out)
        out = self.dropout(out)

        return shortcut + out


class TransformerStream(nn.Module):
    """
    Processes the categorical sequence f_27.
    Components: Embedding -> Learnable Positional Encoding -> Transformer Encoder -> Flatten
    """

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(Config.VOCAB_SIZE, Config.EMBED_DIM)

        # Learnable Positional Embeddings
        # Shape: (1, Seq_Len, Embed_Dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, Config.SEQ_LEN, Config.EMBED_DIM))
        nn.init.normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=Config.EMBED_DIM,
            nhead=Config.TRANSFORMER_HEADS,
            dim_feedforward=Config.EMBED_DIM * 4,
            dropout=Config.TRANSFORMER_DROPOUT,
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=Config.TRANSFORMER_LAYERS
        )

        self.flatten_dim = Config.SEQ_LEN * Config.EMBED_DIM

    def forward(self, x):
        # x shape: (Batch, Seq_Len)
        x = self.embed(x)  # (Batch, Seq, Dim)

        # Add positional encoding (broadcasting over batch)
        x = x + self.pos_embed

        # Pass through Transformer
        x = self.transformer(x)

        # Flatten for fusion
        x = x.reshape(x.size(0), -1)
        return x


class AutoencodingHybridNet(nn.Module):
    """
    Autoencoding Hybrid ResFunnel Network.
    Fuses sequence and continuous data, processes via deep residual backbone,
    and outputs classification logits + auxiliary reconstruction logits.
    """

    def __init__(self):
        super().__init__()

        # --- Stream 1: Categorical Sequence ---
        self.transformer_stream = TransformerStream()

        # --- Fusion ---
        # Transformer output (flattened) + Continuous features (raw)
        fusion_dim = (
            self.transformer_stream.flatten_dim + Config.NUM_CONTINUOUS_FEATURES
        )

        # --- Backbone: Pre-Activation ResFunnel ---
        # Stages: [512, 256, 128]

        # Entry Projection to first stage width
        self.entry_proj = nn.Linear(fusion_dim, Config.BACKBONE_STAGES[0])

        layers = []
        current_dim = Config.BACKBONE_STAGES[0]

        # Construct Stages
        # We use a pattern of [Identity Blocks] -> [Transition Block]
        # Stage 1 (512)
        for _ in range(2):  # 2 Identity blocks
            layers.append(
                ResFunnelBlock(current_dim, current_dim, Config.BACKBONE_DROPOUT)
            )

        # Transition 1->2 (512 -> 256)
        next_dim = Config.BACKBONE_STAGES[1]
        layers.append(ResFunnelBlock(current_dim, next_dim, Config.BACKBONE_DROPOUT))
        current_dim = next_dim

        # Stage 2 (256)
        for _ in range(2):  # 2 Identity blocks
            layers.append(
                ResFunnelBlock(current_dim, current_dim, Config.BACKBONE_DROPOUT)
            )

        # Transition 2->3 (256 -> 128)
        next_dim = Config.BACKBONE_STAGES[2]
        layers.append(ResFunnelBlock(current_dim, next_dim, Config.BACKBONE_DROPOUT))
        current_dim = next_dim

        # Stage 3 (128)
        for _ in range(2):  # 2 Identity blocks
            layers.append(
                ResFunnelBlock(current_dim, current_dim, Config.BACKBONE_DROPOUT)
            )

        self.backbone = nn.Sequential(*layers)

        # --- Heads ---
        final_dim = Config.BACKBONE_STAGES[-1]  # 128

        # 1. Classification Head
        self.cls_head = nn.Linear(final_dim, 1)

        # Removed Auxiliary Reconstruction Head (Cite solution_lesson_node_00065)

    def forward(self, continuous, sequence):
        """
        Args:
            continuous: (Batch, 30) FloatTensor
            sequence: (Batch, 10) LongTensor

        Returns:
            cls_logits: (Batch, 1)
        """
        # Process Streams
        seq_feat = self.transformer_stream(sequence)
        cont_feat = continuous

        # Fusion
        x = torch.cat([seq_feat, cont_feat], dim=1)

        # Backbone
        x = self.entry_proj(x)
        x = self.backbone(x)

        # Classification Output
        cls_logits = self.cls_head(x)

        return cls_logits
