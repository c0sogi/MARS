import torch
import torch.nn as nn
from library.config import Config

# ------------------------------------------------------------------------------
# Model Architecture
# ------------------------------------------------------------------------------


class GatedLinearUnit(nn.Module):
    """
    Gated Linear Unit (GLU) with Sigmoid activation for the gate.
    Structure:
        x -> Linear(in, out * 2) -> split -> a, b
        output = a * sigmoid(b)
    """

    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features * 2)

    def forward(self, x):
        # Project to double dimension
        proj = self.linear(x)
        # Split into value (u) and gate (v) branches
        u, v = proj.chunk(2, dim=-1)
        # Apply Gating
        return u * torch.sigmoid(v)


class ResidualGatedBlock(nn.Module):
    """
    Residual Block with GLU, BatchNorm, and Dropout.
    Structure:
        x_out = x_in + Dropout(BatchNorm(GLU(x_in)))
    Operates at constant width.
    """

    def __init__(self, width, dropout_rate):
        super().__init__()
        self.glu = GatedLinearUnit(width, width)
        self.bn = nn.BatchNorm1d(width)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        residual = x
        out = self.glu(x)
        out = self.bn(out)
        out = self.dropout(out)
        return out + residual


class DownsampleBlock(nn.Module):
    """
    Downsampling Block to reduce dimension between stages.
    Structure:
        Main Path: GLU(in -> out) -> BN -> Dropout
        Skip Path: Linear(in -> out)
        x_out = Main(x) + Skip(x)
    """

    def __init__(self, in_width, out_width, dropout_rate):
        super().__init__()
        # Main path: Project and Gate
        self.glu = GatedLinearUnit(in_width, out_width)
        self.bn = nn.BatchNorm1d(out_width)
        self.dropout = nn.Dropout(dropout_rate)

        # Skip path: Linear Projection to match dimensions
        self.proj = nn.Linear(in_width, out_width)

    def forward(self, x):
        # Skip connection with projection
        residual = self.proj(x)

        # Main path
        out = self.glu(x)
        out = self.bn(out)
        out = self.dropout(out)

        return out + residual


class ResFunnelGLU(nn.Module):
    """
    Residual Funnel Gated Network.
    Features:
        - Categorical Embedding
        - Continuous Feature Fusion
        - 3-Stage Funnel Architecture (512 -> 256 -> 128)
        - Residual Gated Blocks
        - Downsampling with Projections
    """

    def __init__(self):
        super().__init__()

        # 1. Input Processing
        # Embedding for f_27 characters
        self.emb = nn.Embedding(Config.VOCAB_SIZE, Config.EMBED_DIM)

        # Calculate fusion dimension
        # Continuous features + Flattened embeddings (10 chars * 32 dim)
        fusion_input_dim = Config.NUM_CONT_FEATURES + (
            Config.F27_SEQ_LEN * Config.EMBED_DIM
        )

        # Initial projection to Stage 1 width
        self.fusion = nn.Linear(fusion_input_dim, Config.INIT_WIDTH)

        # 2. Backbone Stages
        # Config.STAGES = [512, 256, 128]

        # Stage 1: Width 512
        # Multiple identity blocks (3 blocks)
        self.stage1 = nn.Sequential(
            ResidualGatedBlock(Config.STAGES[0], Config.DROPOUT_RATE),
            ResidualGatedBlock(Config.STAGES[0], Config.DROPOUT_RATE),
            ResidualGatedBlock(Config.STAGES[0], Config.DROPOUT_RATE),
        )

        # Downsample 1: 512 -> 256
        self.down1 = DownsampleBlock(
            Config.STAGES[0], Config.STAGES[1], Config.DROPOUT_RATE
        )

        # Stage 2: Width 256
        # Multiple identity blocks (3 blocks)
        self.stage2 = nn.Sequential(
            ResidualGatedBlock(Config.STAGES[1], Config.DROPOUT_RATE),
            ResidualGatedBlock(Config.STAGES[1], Config.DROPOUT_RATE),
            ResidualGatedBlock(Config.STAGES[1], Config.DROPOUT_RATE),
        )

        # Downsample 2: 256 -> 128
        self.down2 = DownsampleBlock(
            Config.STAGES[1], Config.STAGES[2], Config.DROPOUT_RATE
        )

        # Stage 3: Width 128
        # Multiple identity blocks (3 blocks)
        self.stage3 = nn.Sequential(
            ResidualGatedBlock(Config.STAGES[2], Config.DROPOUT_RATE),
            ResidualGatedBlock(Config.STAGES[2], Config.DROPOUT_RATE),
            ResidualGatedBlock(Config.STAGES[2], Config.DROPOUT_RATE),
        )

        # 3. Output Head
        self.head = nn.Linear(Config.STAGES[2], 1)

    def forward(self, cont_data, cat_data):
        # cont_data: (B, 30)
        # cat_data: (B, 10)

        # Embed and flatten categorical features
        emb = self.emb(cat_data)  # (B, 10, 32)
        emb_flat = emb.view(emb.size(0), -1)  # (B, 320)

        # Concatenate
        x = torch.cat([cont_data, emb_flat], dim=1)

        # Fusion
        x = self.fusion(x)

        # Stage 1
        x = self.stage1(x)

        # Downsample
        x = self.down1(x)

        # Stage 2
        x = self.stage2(x)

        # Downsample
        x = self.down2(x)

        # Stage 3
        x = self.stage3(x)

        # Head (Sigmoid for binary classification)
        return torch.sigmoid(self.head(x))
