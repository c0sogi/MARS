import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from library.config import Config


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output


class SwiGLUBlock(nn.Module):
    """
    Pre-LayerNorm Direct SwiGLU Residual Block.
    Structure: x = x + DropPath(Dropout(Linear_Out(Swish(Gate) * Value)))
    where Gate and Value come from Linear(LayerNorm(x)).
    """

    def __init__(self, dim, drop_path=0.0, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

        # SwiGLU FFN expansion
        # Typically expansion factor is 4, but split into gate and value.
        # To keep parameter count reasonable and follow "Direct" naming,
        # we use an expansion factor of 4 for the hidden dim (standard FFN size).
        hidden_dim = int(dim * 4)

        # Combined projection for gate and value to save one kernel launch
        self.fc_gate_val = nn.Linear(dim, hidden_dim * 2)
        self.fc_out = nn.Linear(hidden_dim, dim)

        self.dropout = nn.Dropout(dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.norm(x)

        # SwiGLU Logic
        gate_val = self.fc_gate_val(x)
        gate, val = gate_val.chunk(2, dim=-1)
        x = F.silu(gate) * val  # SiLU is Swish

        x = self.fc_out(x)
        x = self.dropout(x)
        x = self.drop_path(x)

        return shortcut + x


class HybridSwiGLUNet(nn.Module):
    def __init__(self):
        super().__init__()

        # ----------------------------------------------------------------------
        # Stream 1: Categorical (Post-Norm Transformer)
        # ----------------------------------------------------------------------
        self.seq_len = Config.SEQUENCE_LENGTH
        self.embed_dim = Config.EMBED_DIM

        self.embedding = nn.Embedding(Config.VOCAB_SIZE, self.embed_dim)
        # Learnable Absolute Positional Embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_len, self.embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=Config.TRANSFORMER_HEADS,
            dim_feedforward=self.embed_dim * 4,
            dropout=Config.TRANSFORMER_DROPOUT,
            activation=Config.TRANSFORMER_ACTIVATION,
            norm_first=Config.TRANSFORMER_NORM_FIRST,  # False
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=Config.TRANSFORMER_LAYERS
        )

        # Global Branch Alignment: BatchNorm on flattened output
        transformer_out_dim = self.seq_len * self.embed_dim
        self.cat_bn = nn.BatchNorm1d(transformer_out_dim)

        # ----------------------------------------------------------------------
        # Stream 2: Continuous (Raw)
        # ----------------------------------------------------------------------
        # Just a placeholder for dimension calculation
        cont_dim = Config.NUM_CONTINUOUS_FEATURES

        # ----------------------------------------------------------------------
        # Fusion & Stem
        # ----------------------------------------------------------------------
        fusion_dim = transformer_out_dim + cont_dim
        # Initial backbone width
        current_dim = Config.BACKBONE_STAGES[0]

        self.stem = nn.Linear(fusion_dim, current_dim)

        # ----------------------------------------------------------------------
        # Backbone: LayerNorm SwiGLU ResFunnel
        # ----------------------------------------------------------------------
        self.stages = nn.ModuleList()
        self.transitions = nn.ModuleList()

        # Stochastic Depth Schedule
        total_blocks = sum([Config.BLOCKS_PER_STAGE] * len(Config.BACKBONE_STAGES))
        dpr = [
            x.item()
            for x in torch.linspace(0, Config.STOCHASTIC_DEPTH_MAX, total_blocks)
        ]

        block_idx = 0
        in_dim = current_dim

        for stage_idx, width in enumerate(Config.BACKBONE_STAGES):
            # If not the first stage, add a transition (Downsample)
            if stage_idx > 0:
                self.transitions.append(
                    nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, width))
                )
            else:
                self.transitions.append(nn.Identity())

            # Build Blocks for this stage
            stage_blocks = []
            for _ in range(Config.BLOCKS_PER_STAGE):
                stage_blocks.append(
                    SwiGLUBlock(
                        dim=width,
                        drop_path=dpr[block_idx],
                        dropout=Config.BACKBONE_DROPOUT,
                    )
                )
                block_idx += 1

            self.stages.append(nn.Sequential(*stage_blocks))
            in_dim = width

        # ----------------------------------------------------------------------
        # Output Head
        # ----------------------------------------------------------------------
        self.head = nn.Linear(in_dim, 1)

        # Initialize weights
        self.apply(self._init_weights)

        # Apply specific initializations overriding the default apply
        self._init_specific()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def _init_specific(self):
        # 1. Embeddings: Unit Variance
        nn.init.normal_(self.embedding.weight, std=Config.EMBED_INIT_STD)

        # 2. Positional Embeddings: Low Variance Random Noise
        nn.init.normal_(self.pos_embed, std=Config.POS_EMBED_STD)

        # 3. Transformer: Xavier (Glorot)
        # We iterate through transformer modules to apply Xavier
        for name, p in self.transformer.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # 4. SwiGLU Blocks are handled by _init_weights (Kaiming)
        # 5. Head is handled by _init_weights

    def forward(self, continuous, categorical):
        # Stream 1: Categorical
        # x_cat: (B, SeqLen)
        x_cat = self.embedding(categorical)  # (B, SeqLen, EmbDim)
        x_cat = x_cat + self.pos_embed

        # Transformer Encoder
        x_cat = self.transformer(x_cat)  # (B, SeqLen, EmbDim)

        # Flatten
        B, S, E = x_cat.shape
        x_cat = x_cat.reshape(B, S * E)

        # Global Branch Alignment (BatchNorm)
        x_cat = self.cat_bn(x_cat)

        # Stream 2: Continuous (already normalized)
        x_cont = continuous

        # Fusion
        x = torch.cat([x_cat, x_cont], dim=1)
        x = self.stem(x)

        # Backbone
        for transition, stage in zip(self.transitions, self.stages):
            x = transition(x)
            x = stage(x)

        # Head
        logits = self.head(x)
        return torch.sigmoid(logits)
