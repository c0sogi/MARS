import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class PreActDirectGLUBlock(nn.Module):
    """
    Pre-Activation Direct GLU Residual Block.
    Topology: x_out = x + Dropout(GLU(Linear(BatchNorm(x))))

    This block stabilizes the gating mechanism by normalizing inputs before projection
    and uses a projected residual connection for dimension adaptation.
    """

    def __init__(self, in_features, out_features, dropout_rate):
        super().__init__()
        self.bn = nn.BatchNorm1d(in_features)
        # Direct GLU: Linear projects to 2 * out_features, then GLU halves it
        self.linear = nn.Linear(in_features, out_features * 2)
        self.dropout = nn.Dropout(dropout_rate)

        # Projected Residual Connection: Essential for "ResFunnel" width transitions
        if in_features != out_features:
            self.shortcut = nn.Linear(in_features, out_features)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        # Pre-activation path
        out = self.bn(x)
        out = self.linear(out)
        out = F.glu(out, dim=-1)  # Halves the last dimension
        out = self.dropout(out)

        # Residual addition
        return out + self.shortcut(x)


class HybridModel(nn.Module):
    """
    Optimized Pre-Activation Hybrid Network.

    Architecture:
    1. Transformer Stream: Processes f_27 sequence with GELU activation.
    2. Continuous Stream: Raw normalized features (f_00 - f_30).
    3. Fusion: Concatenation -> Linear Projection (No BN).
    4. Backbone: 3-Stage Pre-Activation Direct GLU ResFunnel (512->256->128).
    5. Head: Minimalist Linear Output.
    """

    def __init__(self):
        super().__init__()

        # ----------------------------------------------------------------------
        # Stream 1: Categorical Sequence (Transformer)
        # ----------------------------------------------------------------------
        self.seq_len = Config.SEQUENCE_LENGTH
        self.embed_dim = Config.EMBED_DIM
        self.vocab_size = 26  # Characters 'A' through 'Z'

        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)

        # Learnable Positional Embeddings (Cite Lesson 30)
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.seq_len, self.embed_dim))

        # Corrected Transformer Encoder (GELU activation, Low Dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=Config.TRANSFORMER_HEADS,
            dim_feedforward=self.embed_dim * 4,
            dropout=Config.TRANSFORMER_DROPOUT,
            activation=Config.TRANSFORMER_ACTIVATION,  # "gelu"
            batch_first=True,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=Config.TRANSFORMER_LAYERS
        )

        # ----------------------------------------------------------------------
        # Fusion Layer
        # ----------------------------------------------------------------------
        # Continuous input dim is 30 (f_00 to f_30, excluding f_27)
        self.cont_dim = 30
        self.seq_flat_dim = self.seq_len * self.embed_dim
        fusion_in_dim = self.cont_dim + self.seq_flat_dim

        backbone_widths = Config.BACKBONE_WIDTHS  # [512, 256, 128]

        # Projection to initial backbone width (No BN before fusion)
        self.fusion_proj = nn.Linear(fusion_in_dim, backbone_widths[0])

        # ----------------------------------------------------------------------
        # Backbone: Pre-Activation Direct GLU ResFunnel
        # ----------------------------------------------------------------------
        layers = []
        current_width = backbone_widths[0]

        # Stage 1: Processing at initial width (512 -> 512)
        layers.append(
            PreActDirectGLUBlock(current_width, current_width, Config.BACKBONE_DROPOUT)
        )

        # Subsequent Stages: Downsampling (512 -> 256, 256 -> 128)
        for width in backbone_widths[1:]:
            layers.append(
                PreActDirectGLUBlock(current_width, width, Config.BACKBONE_DROPOUT)
            )
            current_width = width

        self.backbone = nn.Sequential(*layers)

        # ----------------------------------------------------------------------
        # Output Head
        # ----------------------------------------------------------------------
        # Minimalist design: Single Linear layer, no extra dropout/BN
        self.head = nn.Linear(current_width, 1)

        self._init_weights()

    def _init_weights(self):
        """
        Xavier Uniform initialization for linear layers.
        Normal initialization for positional embeddings.
        """
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # Initialize positional embeddings
        nn.init.normal_(self.pos_embedding, mean=0, std=0.02)

    def forward(self, continuous, sequence):
        # --- Stream 1: Sequence ---
        # sequence: (Batch, 10)
        x_seq = self.embedding(sequence)  # (Batch, 10, 32)
        x_seq = x_seq + self.pos_embedding
        x_seq = self.transformer_encoder(x_seq)
        x_seq = x_seq.flatten(start_dim=1)  # (Batch, 320)

        # --- Stream 2: Continuous ---
        # continuous: (Batch, 30) - Raw normalized features

        # --- Fusion ---
        x_fused = torch.cat([x_seq, continuous], dim=1)
        x = self.fusion_proj(x_fused)

        # --- Backbone ---
        x = self.backbone(x)

        # --- Head ---
        logits = self.head(x)
        return logits
