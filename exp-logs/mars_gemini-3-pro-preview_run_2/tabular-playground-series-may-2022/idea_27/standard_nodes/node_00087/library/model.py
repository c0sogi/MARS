import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath


class GLUBlock(nn.Module):
    """
    Pre-Activation Direct GLU Residual Block with Stochastic Depth.
    Structure: x + DropPath(Dropout(GLU(Linear(BN(x)))))
    """

    def __init__(self, in_dim, drop_path=0.0, dropout=0.0):
        super().__init__()
        self.norm = nn.BatchNorm1d(in_dim)
        # GLU reduces dim by half, so we project to 2 * in_dim
        self.linear = nn.Linear(in_dim, in_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # Init Linear for GLU: Xavier Uniform (Glorot)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        identity = x
        out = self.norm(x)
        out = self.linear(out)
        out = F.glu(out, dim=-1)  # Halves dimension back to in_dim
        out = self.dropout(out)
        out = self.drop_path(out)
        return identity + out


class TransformerStream(nn.Module):
    """
    Stream 1: Categorical Sequence processing using GELU-Transformer.
    """

    def __init__(
        self, vocab_size, embed_dim, seq_len, num_layers, num_heads, dropout, activation
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Learnable Positional Encoding
        self.pos_encoder = nn.Parameter(torch.zeros(1, seq_len, embed_dim))
        # Init Pos Enc with Low Variance Noise
        nn.init.normal_(self.pos_encoder, mean=0.0, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # x: (B, L)
        x = self.embedding(x)  # (B, L, D)
        x = x + self.pos_encoder
        x = self.transformer(x)
        x = x.flatten(1)  # (B, L*D)
        return x


class StochasticResFunnelHybrid(nn.Module):
    """
    Hybrid architecture fusing a Transformer stream and raw continuous stream
    into a Stochastic-Regularized ResFunnel backbone.
    """

    def __init__(self, config, vocab_size):
        super().__init__()

        # --- Stream 1: Categorical Sequence ---
        self.stream1 = TransformerStream(
            vocab_size=vocab_size,
            embed_dim=config.EMBED_DIM,
            seq_len=config.SEQ_LEN,
            num_layers=config.TRANSFORMER_LAYERS,
            num_heads=config.TRANSFORMER_HEADS,
            dropout=config.TRANSFORMER_DROPOUT,
            activation=config.TRANSFORMER_ACTIVATION,
        )
        self.stream1_dim = config.SEQ_LEN * config.EMBED_DIM

        # --- Stream 2: Continuous ---
        # 30 continuous features
        self.stream2_dim = 30

        # --- Fusion ---
        fusion_in_dim = self.stream1_dim + self.stream2_dim
        # Linear Stem: Decouples alignment from first block. No BN here.
        self.stem = nn.Linear(fusion_in_dim, config.STEM_DIM)

        # --- Backbone ---
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        current_dim = config.STEM_DIM

        # Calculate total blocks for stochastic depth schedule
        total_blocks = len(config.BACKBONE_STAGES) * config.BLOCKS_PER_STAGE
        global_block_idx = 0

        for i, stage_dim in enumerate(config.BACKBONE_STAGES):
            # Downsampling / Transition
            if i > 0:
                # Projected Residual Connection for downsampling (BN -> Linear)
                downsample = nn.Sequential(
                    nn.BatchNorm1d(current_dim), nn.Linear(current_dim, stage_dim)
                )
                self.downsamples.append(downsample)
            else:
                # First stage transition (usually Identity if dims match)
                if current_dim != stage_dim:
                    self.downsamples.append(nn.Linear(current_dim, stage_dim))
                else:
                    self.downsamples.append(nn.Identity())

            # Blocks
            blocks = nn.ModuleList()
            for _ in range(config.BLOCKS_PER_STAGE):
                # Linear DropPath schedule: 0.0 -> DROP_PATH_MAX
                dp_rate = config.DROP_PATH_MAX * (global_block_idx / total_blocks)
                blocks.append(
                    GLUBlock(stage_dim, drop_path=dp_rate, dropout=config.BLOCK_DROPOUT)
                )
                global_block_idx += 1

            self.stages.append(blocks)
            current_dim = stage_dim

        # --- Head ---
        # Minimalist design: Single Linear layer
        self.head = nn.Linear(current_dim, 1)

    def forward(self, x_seq, x_cont):
        # Stream 1
        x_s = self.stream1(x_seq)

        # Stream 2
        x_c = x_cont  # (B, 30)

        # Fusion
        x = torch.cat([x_s, x_c], dim=1)
        x = self.stem(x)

        # Backbone
        for downsample, stage_blocks in zip(self.downsamples, self.stages):
            x = downsample(x)
            for block in stage_blocks:
                x = block(x)

        # Head
        logits = self.head(x)
        return logits


def get_model(config, vocab_size):
    return StochasticResFunnelHybrid(config, vocab_size)
