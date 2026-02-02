import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class GatedGLU(nn.Module):
    """
    Gated Linear Unit component: Linear -> GLU (Sigmoid) -> Linear.
    Implements the noise-filtering mechanism described in the idea.

    Structure:
    1. Projection: Input (d_in) -> Hidden (2 * d_hidden)
    2. Gating: Split into value and gate. Output = value * Sigmoid(gate).
    3. Projection: Hidden (d_hidden) -> Output (d_out)
    """

    def __init__(self, in_features, hidden_features, out_features, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features * 2)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (Batch, ..., in_features)

        # 1. Project to double hidden dimension
        x = self.fc1(x)

        # 2. Split for GLU
        x, gate = x.chunk(2, dim=-1)

        # 3. Gated Activation (Sigmoid)
        x = x * torch.sigmoid(gate)

        # 4. Dropout
        x = self.dropout(x)

        # 5. Project to output
        x = self.fc2(x)

        return x


class GatedTransformerBlock(nn.Module):
    """
    Transformer block where the standard FFN is replaced by a GatedGLU network.
    Uses Pre-Norm architecture for stability.
    """

    def __init__(self, embed_dim, num_heads, ffn_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.gated_ffn = GatedGLU(embed_dim, ffn_dim, embed_dim, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # Self-Attention Block
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.dropout1(attn_out)

        # Gated FFN Block
        x_norm = self.norm2(x)
        ffn_out = self.gated_ffn(x_norm)
        x = x + self.dropout2(ffn_out)

        return x


class ResFunnelBlock(nn.Module):
    """
    Residual Gated Block for the backbone.
    Supports dimension changes via Projected Residual Connections.
    """

    def __init__(self, in_features, out_features, dropout=0.35):
        super().__init__()
        self.norm = nn.LayerNorm(in_features)

        # We use an expansion factor of 2 for the internal GLU state
        hidden_dim = out_features * 2

        self.gated_block = GatedGLU(
            in_features, hidden_dim, out_features, dropout=dropout
        )

        # Projected Residual Connection
        if in_features != out_features:
            self.shortcut = nn.Linear(in_features, out_features)
        else:
            self.shortcut = nn.Identity()

        self.dropout_residual = nn.Dropout(dropout)

    def forward(self, x):
        # Main Branch
        residual = x

        # Norm -> Gated Block
        out = self.norm(x)
        out = self.gated_block(out)
        out = self.dropout_residual(out)

        # Residual Branch (Projected if dims differ)
        res = self.shortcut(residual)

        return out + res


class GatedTransformerResFunnelHybrid(nn.Module):
    """
    The main architecture fusing a Gated Transformer Encoder for categorical sequences
    and a ResFunnel-GLU backbone for the combined features.
    """

    def __init__(self):
        super().__init__()

        # --- Stream 1: Gated Transformer Categorical Encoder ---
        self.embed_dim = Config.EMBED_DIM
        self.seq_len = Config.SEQUENCE_LENGTH

        # Token Embeddings
        self.embedding = nn.Embedding(Config.VOCAB_SIZE, self.embed_dim)

        # Learnable Positional Embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, self.seq_len, self.embed_dim))

        # Transformer Blocks
        self.transformer_blocks = nn.ModuleList(
            [
                GatedTransformerBlock(
                    embed_dim=self.embed_dim,
                    num_heads=Config.TRANSFORMER_HEADS,
                    ffn_dim=Config.TRANSFORMER_FFN_DIM,
                    dropout=Config.TRANSFORMER_DROPOUT,
                )
                for _ in range(Config.TRANSFORMER_LAYERS)
            ]
        )

        # Flattened dimension: 10 tokens * 32 dim = 320
        self.transformer_out_dim = self.seq_len * self.embed_dim

        # --- Stream 2: Raw Continuous Features ---
        # No processing layers, just input dimension definition
        self.continuous_dim = Config.NUM_CONTINUOUS_FEATURES

        # --- Fusion & Backbone ---
        fusion_dim = self.transformer_out_dim + self.continuous_dim

        # ResFunnel Backbone
        # We construct stages based on Config.BACKBONE_WIDTHS [512, 256, 128]
        # Strategy: Transition Block (Change Dim) -> Identity Block (Depth)

        layers = []
        current_dim = fusion_dim

        for width in Config.BACKBONE_WIDTHS:
            # Transition / Projection Block
            layers.append(
                ResFunnelBlock(current_dim, width, dropout=Config.BACKBONE_DROPOUT)
            )
            # Identity Block for depth
            layers.append(ResFunnelBlock(width, width, dropout=Config.BACKBONE_DROPOUT))
            current_dim = width

        self.backbone = nn.Sequential(*layers)

        # --- Head ---
        self.head = nn.Linear(current_dim, 1)

    def forward(self, continuous, categorical):
        """
        Args:
            continuous: (Batch, 30) FloatTensor
            categorical: (Batch, 10) LongTensor (Token indices)
        """
        # --- Stream 1: Transformer ---
        # Embedding: (B, 10, 32)
        x_cat = self.embedding(categorical)

        # Add Positional Encoding
        x_cat = x_cat + self.pos_embedding

        # Apply Transformer Blocks
        for block in self.transformer_blocks:
            x_cat = block(x_cat)

        # Flatten: (B, 320)
        x_cat_flat = x_cat.reshape(x_cat.size(0), -1)

        # --- Fusion ---
        # Concatenate flattened sequence features with raw continuous features
        # (B, 320 + 30) = (B, 350)
        x_fused = torch.cat([x_cat_flat, continuous], dim=1)

        # --- Backbone ---
        features = self.backbone(x_fused)

        # --- Head ---
        logits = self.head(features)

        # Return logits (B, 1). Loss function (BCEWithLogits) handles sigmoid.
        return logits
