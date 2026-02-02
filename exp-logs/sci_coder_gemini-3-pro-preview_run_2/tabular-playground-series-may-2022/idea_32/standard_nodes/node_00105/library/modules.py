import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from library.config import Config


class SwiGLU(nn.Module):
    """
    SwiGLU activation: Split input into two halves, apply Swish (SiLU) to one, and multiply.
    Input: (..., 2 * dim)
    Output: (..., dim)
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return x1 * F.silu(x2)


class LearnablePositionalEncoding(nn.Module):
    """
    Learnable Absolute Positional Embeddings.
    Initialized with Low Variance Random Noise N(0, 0.02).
    """

    def __init__(self, max_len, d_model, std=0.02):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, d_model))
        self.std = std
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.pos_embed, mean=0.0, std=self.std)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        # Broadcasting the batch dimension
        return x + self.pos_embed[:, :seq_len, :]


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample.
    """

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        # Work with any number of dimensions, broadcasting the batch dim
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output


class DirectSwiGLUResidualBlock(nn.Module):
    """
    Pre-LayerNorm Direct SwiGLU Residual Block.
    Structure: x + DropPath(Dropout(SwiGLU(Linear(LayerNorm(x)))))
    """

    def __init__(self, dim, dropout=0.0, drop_path=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        # Linear projects to 2*dim because SwiGLU consumes half for gating
        self.linear = nn.Linear(dim, 2 * dim)
        self.swiglu = SwiGLU()
        self.dropout = nn.Dropout(dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        # Kaiming Uniform for the Linear layer
        nn.init.kaiming_uniform_(self.linear.weight, a=math.sqrt(5))
        if self.linear.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.linear.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.linear.bias, -bound, bound)

    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        x = self.linear(x)
        x = self.swiglu(x)
        x = self.dropout(x)
        x = self.drop_path(x)
        return shortcut + x


class HybridResFunnelModel(nn.Module):
    """
    Dual-Stem Post-Norm SwiGLU-ResFunnel Network.
    """

    def __init__(self, config=None):
        super().__init__()
        self.cfg = config if config else Config

        # ----------------------------------------------------------------------
        # Stream 1: Categorical Sequence (Post-Norm Transformer)
        # ----------------------------------------------------------------------
        self.char_embed = nn.Embedding(self.cfg.VOCAB_SIZE, self.cfg.EMBED_DIM)
        self.pos_embed = LearnablePositionalEncoding(
            self.cfg.SEQ_LEN, self.cfg.EMBED_DIM, std=self.cfg.POS_EMBED_STD
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.cfg.EMBED_DIM,
            nhead=self.cfg.TRANSFORMER_HEADS,
            dim_feedforward=self.cfg.EMBED_DIM * 4,
            dropout=self.cfg.TRANSFORMER_DROPOUT,
            activation=self.cfg.TRANSFORMER_ACTIVATION,
            batch_first=True,
            norm_first=self.cfg.TRANSFORMER_NORM_FIRST,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=self.cfg.TRANSFORMER_LAYERS
        )

        # Flattened dimension: 10 tokens * 32 dim = 320
        self.flat_seq_dim = self.cfg.SEQ_LEN * self.cfg.EMBED_DIM

        # Sequence Stem: Linear -> LN -> SwiGLU -> Dropout
        # Projects flattened sequence to STEM_DIM
        self.seq_stem_linear = nn.Linear(self.flat_seq_dim, 2 * self.cfg.STEM_DIM)
        self.seq_stem_norm = nn.LayerNorm(2 * self.cfg.STEM_DIM)
        self.seq_stem_swiglu = SwiGLU()
        self.seq_stem_dropout = nn.Dropout(self.cfg.MAIN_DROPOUT)

        # ----------------------------------------------------------------------
        # Stream 2: Continuous Preservation
        # ----------------------------------------------------------------------
        # Continuous Stem: Linear -> LN -> SwiGLU -> Dropout
        # Projects raw continuous features to STEM_DIM
        self.cont_stem_linear = nn.Linear(
            self.cfg.NUM_CONT_FEATURES, 2 * self.cfg.STEM_DIM
        )
        self.cont_stem_norm = nn.LayerNorm(2 * self.cfg.STEM_DIM)
        self.cont_stem_swiglu = SwiGLU()
        self.cont_stem_dropout = nn.Dropout(self.cfg.MAIN_DROPOUT)

        # ----------------------------------------------------------------------
        # Fusion & Backbone (SwiGLU ResFunnel)
        # ----------------------------------------------------------------------
        # Concatenate outputs of both stems
        fusion_dim = self.cfg.STEM_DIM * 2

        # Project to first stage width
        self.fusion_proj = nn.Linear(fusion_dim, self.cfg.BACKBONE_STAGES[0])

        self.backbone = nn.ModuleList()
        current_dim = self.cfg.BACKBONE_STAGES[0]

        total_blocks = len(self.cfg.BACKBONE_STAGES) * self.cfg.BLOCKS_PER_STAGE
        block_idx = 0

        for i, stage_dim in enumerate(self.cfg.BACKBONE_STAGES):
            # If dimension changes between stages (funneling down)
            if i > 0 and stage_dim != self.cfg.BACKBONE_STAGES[i - 1]:
                self.backbone.append(
                    nn.Linear(self.cfg.BACKBONE_STAGES[i - 1], stage_dim)
                )
                current_dim = stage_dim

            stage_blocks = nn.Sequential()
            for _ in range(self.cfg.BLOCKS_PER_STAGE):
                # Linear Stochastic Depth Schedule
                sd_prob = self.cfg.STOCHASTIC_DEPTH_MIN + (
                    self.cfg.STOCHASTIC_DEPTH_MAX - self.cfg.STOCHASTIC_DEPTH_MIN
                ) * (block_idx / (total_blocks - 1))

                stage_blocks.append(
                    DirectSwiGLUResidualBlock(
                        dim=current_dim,
                        dropout=self.cfg.MAIN_DROPOUT,
                        drop_path=sd_prob,
                    )
                )
                block_idx += 1
            self.backbone.append(stage_blocks)

        # ----------------------------------------------------------------------
        # Output Head
        # ----------------------------------------------------------------------
        self.head = nn.Linear(self.cfg.BACKBONE_STAGES[-1], 1)

        self._init_weights()

    def _init_weights(self):
        # Transformer Initialization (Xavier)
        for p in self.transformer_encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # Stems Initialization (Kaiming)
        nn.init.kaiming_uniform_(self.seq_stem_linear.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.cont_stem_linear.weight, a=math.sqrt(5))

        # Fusion Projection Initialization
        nn.init.kaiming_uniform_(self.fusion_proj.weight, a=math.sqrt(5))

        # Head Initialization
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, continuous, sequence):
        # --- Stream 1 ---
        # sequence: (B, 10) -> (B, 10, 32)
        x_seq = self.char_embed(sequence)
        x_seq = self.pos_embed(x_seq)
        x_seq = self.transformer_encoder(x_seq)
        x_seq = x_seq.flatten(1)  # (B, 320)

        # Sequence Stem
        x_seq = self.seq_stem_linear(x_seq)
        x_seq = self.seq_stem_norm(x_seq)
        x_seq = self.seq_stem_swiglu(x_seq)
        x_seq = self.seq_stem_dropout(x_seq)  # (B, STEM_DIM)

        # --- Stream 2 ---
        # Continuous Stem
        x_cont = self.cont_stem_linear(continuous)
        x_cont = self.cont_stem_norm(x_cont)
        x_cont = self.cont_stem_swiglu(x_cont)
        x_cont = self.cont_stem_dropout(x_cont)  # (B, STEM_DIM)

        # --- Fusion ---
        x = torch.cat([x_seq, x_cont], dim=1)  # (B, 2*STEM_DIM)
        x = self.fusion_proj(x)

        # --- Backbone ---
        for layer in self.backbone:
            x = layer(x)

        # --- Head ---
        logits = self.head(x)
        return logits
