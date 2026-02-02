import torch
import torch.nn as nn
import torch.nn.functional as F


class DualViewEncoder(nn.Module):
    """
    Encodes categorical sequence data into two views:
    1. Contextual View: Processed by a Transformer to capture dependencies.
    2. Local View: Raw embeddings flattened to preserve exact identity/position.
    """

    def __init__(
        self,
        num_embeddings=32,
        embedding_dim=32,
        seq_len=10,
        nhead=4,
        num_layers=2,
        dropout=0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)

        # Positional Encoding for Contextual View
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, embedding_dim))
        nn.init.normal_(self.pos_embedding, mean=0.0, std=0.02)

        # Transformer Encoder for Contextual View
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # x: (Batch, Seq_Len)

        # Shared Embedding
        emb = self.embedding(x)  # (B, L, D)

        # View 1: Contextual
        x_ctx = emb + self.pos_embedding
        x_ctx = self.transformer(x_ctx)
        x_ctx_flat = x_ctx.reshape(x_ctx.size(0), -1)  # Flatten

        # View 2: Local
        x_loc_flat = emb.reshape(emb.size(0), -1)  # Flatten directly

        return x_ctx_flat, x_loc_flat


class ResidualGatedBlock(nn.Module):
    """
    A residual block with Gated Linear Unit (GLU), Batch Normalization, and Dropout.
    Supports dimension changes via projected residual connection.
    """

    def __init__(self, in_dim, out_dim, dropout=0.35):
        super().__init__()

        self.bn = nn.BatchNorm1d(in_dim)

        # GLU splits the output in half, so we need 2 * out_dim
        self.linear = nn.Linear(in_dim, out_dim * 2)
        self.dropout = nn.Dropout(dropout)

        # Projected Residual Connection
        if in_dim != out_dim:
            self.project = nn.Linear(in_dim, out_dim)
        else:
            self.project = nn.Identity()

    def forward(self, x):
        shortcut = self.project(x)

        out = self.bn(x)
        out = self.linear(out)
        out = F.glu(out, dim=-1)  # Applies Sigmoid to one half and multiplies
        out = self.dropout(out)

        return shortcut + out


class ResFunnelBackbone(nn.Module):
    """
    A deep residual backbone with a funnel topology (decreasing width).
    """

    def __init__(self, input_dim, dims=[512, 256, 128], dropout=0.35):
        super().__init__()

        # Initial Projection
        self.proj = nn.Linear(input_dim, dims[0])

        # Build Stages
        self.stages = nn.ModuleList()

        # Stage 1 (512)
        # Stacked blocks at initial dimension
        self.stages.append(
            nn.Sequential(
                ResidualGatedBlock(dims[0], dims[0], dropout),
                ResidualGatedBlock(dims[0], dims[0], dropout),
            )
        )

        # Transition 1 -> 2 (512 -> 256)
        self.stages.append(ResidualGatedBlock(dims[0], dims[1], dropout))

        # Stage 2 (256)
        self.stages.append(
            nn.Sequential(
                ResidualGatedBlock(dims[1], dims[1], dropout),
                ResidualGatedBlock(dims[1], dims[1], dropout),
            )
        )

        # Transition 2 -> 3 (256 -> 128)
        self.stages.append(ResidualGatedBlock(dims[1], dims[2], dropout))

        # Stage 3 (128)
        self.stages.append(
            nn.Sequential(
                ResidualGatedBlock(dims[2], dims[2], dropout),
                ResidualGatedBlock(dims[2], dims[2], dropout),
            )
        )

        self.output_dim = dims[-1]

    def forward(self, x):
        x = self.proj(x)
        for stage in self.stages:
            x = stage(x)
        return x


class DualViewHybridResFunnel(nn.Module):
    """
    The main architecture fusing Dual-View Categorical Embeddings and Continuous Features
    into a ResFunnel Backbone.
    """

    def __init__(
        self,
        num_continuous=30,
        vocab_size=32,
        embedding_dim=32,
        seq_len=10,
        transformer_layers=2,
        backbone_dims=[512, 256, 128],
        dropout=0.35,
    ):
        super().__init__()

        # 1. Dual-View Encoder
        self.encoder = DualViewEncoder(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            seq_len=seq_len,
            num_layers=transformer_layers,
        )

        # Calculate Fusion Dimension
        # Contextual View (Flattened) + Local View (Flattened) + Continuous
        fusion_dim = (
            (seq_len * embedding_dim) + (seq_len * embedding_dim) + num_continuous
        )

        # 2. Backbone
        self.backbone = ResFunnelBackbone(
            input_dim=fusion_dim, dims=backbone_dims, dropout=dropout
        )

        # 3. Output Head
        self.head = nn.Linear(self.backbone.output_dim, 1)

    def forward(self, x_cont, x_cat):
        """
        Args:
            x_cont (Tensor): Continuous features (Batch, 30)
            x_cat (Tensor): Categorical features (Batch, 10)
        """
        # Encode Categoricals
        ctx_flat, loc_flat = self.encoder(x_cat)

        # Fusion
        fused = torch.cat([ctx_flat, loc_flat, x_cont], dim=1)

        # Backbone
        features = self.backbone(fused)

        # Output
        logits = self.head(features)
        return torch.sigmoid(logits)
