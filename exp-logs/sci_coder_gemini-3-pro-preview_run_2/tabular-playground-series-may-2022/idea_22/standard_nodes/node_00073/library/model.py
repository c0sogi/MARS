import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config
from library.utils import kaiming_init_weights


class PreActGLUBlock(nn.Module):
    """
    Pre-Activation Residual Block with GLU activation.
    Structure: x + Dropout(GLU(Linear(BatchNorm(x))))

    Handles dimension changes via a Projected Residual Connection on the skip path.
    """

    def __init__(self, in_features, out_features, dropout_rate):
        super().__init__()
        self.bn = nn.BatchNorm1d(in_features)
        self.linear = nn.Linear(in_features, out_features * 2)  # *2 for GLU splitting
        self.dropout = nn.Dropout(dropout_rate)

        # Projected Residual Connection if dimensions change
        if in_features != out_features:
            self.skip = nn.Linear(in_features, out_features)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        # Pre-Activation: BN first
        out = self.bn(x)

        # Main Branch
        out = self.linear(out)
        out = F.glu(out, dim=-1)  # Halves dimension to out_features
        out = self.dropout(out)

        # Skip Connection
        residual = self.skip(x)

        return residual + out


class TransformerStream(nn.Module):
    """
    Processes the sequence feature (f_27) using a GELU-Transformer.
    """

    def __init__(
        self, vocab_size, embed_dim, seq_len, num_heads, num_layers, dropout, activation
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # Learnable Positional Embeddings
        # Initialize with random noise to break symmetry (Cite solution_lesson_node_00072)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, embed_dim))

        # Standard Transformer Encoder
        # activation="gelu" as per Lesson 68
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=False,  # Standard Encoder
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.flatten_dim = seq_len * embed_dim

    def forward(self, x):
        # x: (Batch, Seq_Len)
        x = self.embedding(x)  # (Batch, Seq_Len, Embed_Dim)
        x = x + self.pos_embed  # Add positional info
        x = self.encoder(x)
        x = x.reshape(x.size(0), -1)  # Flatten: (Batch, Seq_Len * Embed_Dim)
        return x


class SustainedDepthHybridNet(nn.Module):
    """
    Sustained-Depth Pre-Activation Hybrid Network.
    Fuses a Transformer stream for sequences with raw continuous features,
    processed by a deep Pre-Act ResFunnel backbone.
    """

    def __init__(
        self,
        num_cont_features=Config.NUM_CONT_FEATURES,
        vocab_size=Config.VOCAB_SIZE,
        embed_dim=Config.EMBED_DIM,
        seq_len=Config.SEQ_LEN,
        transformer_layers=Config.TRANSFORMER_LAYERS,
        transformer_heads=Config.TRANSFORMER_HEADS,
        transformer_activation=Config.TRANSFORMER_ACTIVATION,
        transformer_dropout=Config.DROPOUT_TRANSFORMER,
        backbone_stages=Config.BACKBONE_STAGES,
        blocks_per_stage=Config.BLOCKS_PER_STAGE,
        backbone_dropout=Config.DROPOUT_BACKBONE,
    ):
        super().__init__()

        # Stream 1: Sequence Data
        self.transformer = TransformerStream(
            vocab_size,
            embed_dim,
            seq_len,
            transformer_heads,
            transformer_layers,
            transformer_dropout,
            transformer_activation,
        )

        # Calculate Fusion Dimension
        # Stream 2 (Continuous) is concatenated raw
        fusion_dim = self.transformer.flatten_dim + num_cont_features

        # Backbone: Sustained-Depth Pre-Activation ResFunnel
        layers = []
        current_dim = fusion_dim

        for stage_width in backbone_stages:
            # For each stage, we stack 'blocks_per_stage' blocks.
            # The first block handles the transition (current_dim -> stage_width).
            # Subsequent blocks are constant width (stage_width -> stage_width).

            for i in range(blocks_per_stage):
                in_d = current_dim if i == 0 else stage_width
                out_d = stage_width

                layers.append(PreActGLUBlock(in_d, out_d, backbone_dropout))

            current_dim = stage_width

        self.backbone = nn.Sequential(*layers)

        # Output Head: Minimalist
        # Single Linear layer, no extra BN/Dropout
        self.head = nn.Linear(backbone_stages[-1], 1)

        # Initialization
        # Apply Kaiming only to backbone and head, protecting Transformer defaults (Cite solution_lesson_node_00072)
        self.backbone.apply(kaiming_init_weights)
        self.head.apply(kaiming_init_weights)

    def forward(self, cont_data, seq_data):
        """
        Args:
            cont_data: (Batch, 30) Normalized continuous features.
            seq_data: (Batch, 10) Integer encoded sequence features.
        """
        # 1. Process Sequence Stream
        seq_features = self.transformer(seq_data)

        # 2. Fusion
        # Concatenate flattened sequence features with raw continuous features
        # No BN/Projection here; passed directly to backbone
        fused = torch.cat([seq_features, cont_data], dim=1)

        # 3. Backbone
        # The first block's BN will handle distribution alignment
        features = self.backbone(fused)

        # 4. Output Head
        logits = self.head(features)

        return logits.squeeze(-1)
