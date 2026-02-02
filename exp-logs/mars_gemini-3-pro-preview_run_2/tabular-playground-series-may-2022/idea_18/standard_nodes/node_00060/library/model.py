import torch
import torch.nn as nn
from library.config import Config


class TransformerStream(nn.Module):
    """
    Stream 1: Categorical Sequence (Transformer)
    Processes the tokenized character sequence from f_27.
    """

    def __init__(self):
        super().__init__()
        # Embedding Layer
        self.embedding = nn.Embedding(Config.VOCAB_SIZE, Config.EMBEDDING_DIM)

        # Learnable Positional Embeddings
        # Shape: (1, SEQ_LEN, EMBEDDING_DIM)
        self.pos_encoder = nn.Parameter(
            torch.zeros(1, Config.SEQ_LEN, Config.EMBEDDING_DIM)
        )
        nn.init.normal_(self.pos_encoder, mean=0.0, std=0.02)

        # Standard Transformer Encoder
        # Using batch_first=True for (Batch, Seq, Feature)
        # Feedforward dim is typically 4x embedding dim
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=Config.EMBEDDING_DIM,
            nhead=Config.TRANSFORMER_HEADS,
            dim_feedforward=Config.EMBEDDING_DIM * 4,
            dropout=Config.TRANSFORMER_DROPOUT,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=Config.TRANSFORMER_LAYERS
        )

        # Output dimension after flattening
        self.flatten_dim = Config.SEQ_LEN * Config.EMBEDDING_DIM

    def forward(self, x):
        # x shape: (batch_size, seq_len)

        # Embed
        x = self.embedding(x)  # (batch, seq, dim)

        # Add Positional Encoding
        x = x + self.pos_encoder

        # Encode
        x = self.transformer_encoder(x)

        # Flatten
        x = x.reshape(x.size(0), -1)
        return x


class PreActGLUBlock(nn.Module):
    """
    Pre-Activation Direct GLU Residual Block.
    Topology: x_out = x + Dropout(GLU(Linear(BatchNorm(x))))
    """

    def __init__(self, dim, dropout_rate):
        super().__init__()
        self.bn = nn.BatchNorm1d(dim)
        # GLU halves the dimension, so Linear output must be 2 * dim
        self.linear = nn.Linear(dim, dim * 2)
        self.glu = nn.GLU(dim=1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        identity = x

        # Pre-Activation
        out = self.bn(x)
        out = self.linear(out)
        out = self.glu(out)
        out = self.dropout(out)

        return identity + out


class ProjectedTransition(nn.Module):
    """
    Transition layer between stages with decreasing width.
    Uses Projected Residual Connections to strictly preserve gradient flow.
    """

    def __init__(self, in_dim, out_dim, dropout_rate):
        super().__init__()

        # Shortcut path: Linear Projection to match dimensions
        self.project = nn.Linear(in_dim, out_dim)

        # Main path: BN -> Linear -> GLU -> Dropout
        self.bn = nn.BatchNorm1d(in_dim)
        self.main_linear = nn.Linear(in_dim, out_dim * 2)
        self.glu = nn.GLU(dim=1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        shortcut = self.project(x)

        out = self.bn(x)
        out = self.main_linear(out)
        out = self.glu(out)
        out = self.dropout(out)

        return shortcut + out


class HybridResFunnel(nn.Module):
    """
    Hybrid ResFunnel Network.
    Fuses categorical transformer features with raw continuous features,
    processes via a 3-stage residual backbone, and uses the final compressed representation.
    Cite solution_lesson_node_00059: Enforcing a strict information bottleneck is superior to multi-scale aggregation.
    """

    def __init__(self):
        super().__init__()

        # --- Stream 1: Categorical ---
        self.transformer_stream = TransformerStream()

        # --- Fusion Layer ---
        # Concatenate Flattened Transformer + Raw Continuous
        fusion_in_dim = self.transformer_stream.flatten_dim + Config.NUM_CONT_FEATURES
        self.fusion_project = nn.Linear(fusion_in_dim, Config.INITIAL_WIDTH)

        # --- Backbone: Pre-Activation ResFunnel ---
        # We use 2 blocks per stage as a standard depth configuration
        blocks_per_stage = 2
        dropout = Config.BACKBONE_DROPOUT
        stages = Config.BACKBONE_STAGES  # [512, 256, 128]

        # Stage 1 (Width 512)
        self.stage1_blocks = nn.Sequential(
            *[PreActGLUBlock(stages[0], dropout) for _ in range(blocks_per_stage)]
        )

        # Transition 1 -> 2
        self.trans1 = ProjectedTransition(stages[0], stages[1], dropout)

        # Stage 2 (Width 256)
        self.stage2_blocks = nn.Sequential(
            *[PreActGLUBlock(stages[1], dropout) for _ in range(blocks_per_stage)]
        )

        # Transition 2 -> 3
        self.trans2 = ProjectedTransition(stages[1], stages[2], dropout)

        # Stage 3 (Width 128)
        self.stage3_blocks = nn.Sequential(
            *[PreActGLUBlock(stages[2], dropout) for _ in range(blocks_per_stage)]
        )

        # --- Head: Standard Classification ---
        # Uses output from final stage (128)
        final_dim = stages[-1]

        self.head = nn.Sequential(
            nn.BatchNorm1d(final_dim),
            nn.Dropout(dropout),
            nn.Linear(final_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, cont_data, cat_data):
        """
        Args:
            cont_data: Tensor (batch, 30)
            cat_data: Tensor (batch, 10)
        """
        # 1. Process Categorical Stream
        cat_feat = self.transformer_stream(cat_data)

        # 2. Fusion
        # Concatenate raw continuous features
        fused = torch.cat([cat_feat, cont_data], dim=1)
        x = self.fusion_project(fused)

        # 3. Backbone (Funnel)

        # Stage 1
        x = self.stage1_blocks(x)
        x = self.trans1(x)

        # Stage 2
        x = self.stage2_blocks(x)
        x = self.trans2(x)

        # Stage 3
        x = self.stage3_blocks(x)

        # 4. Classification
        return self.head(x)
