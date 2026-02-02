import torch
import torch.nn as nn
import torch.nn.functional as F
import library.config as config


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        # Work with any number of dimensions, assuming batch is dim 0
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output


class SwiGLUBlock(nn.Module):
    """
    Pre-LayerNorm Direct SwiGLU Residual Block.
    Structure: x_out = x + DropPath(Dropout(SwiGLU(Linear(LayerNorm(x)))))
    """

    def __init__(self, dim, drop_path=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        # Linear projects to 2*dim to support SwiGLU split
        self.linear = nn.Linear(dim, 2 * dim)
        self.dropout = nn.Dropout(config.BACKBONE_DROPOUT)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input_x = x

        # Pre-Norm
        x = self.norm(x)

        # Linear Projection
        x = self.linear(x)

        # SwiGLU Activation
        # Split into value and gate
        x1, x2 = x.chunk(2, dim=-1)
        # Swish(x1) * x2
        x = F.silu(x1) * x2

        # Regularization
        x = self.dropout(x)
        x = self.drop_path(x)

        # Residual connection
        return input_x + x


class ManufacturingNet(nn.Module):
    def __init__(self):
        super().__init__()

        # ----------------------------------------------------------------------
        # Stream 1: Categorical Sequence (Position-Distinct)
        # ----------------------------------------------------------------------
        # Vocab size covers all positions (26 chars * 10 positions = 260)
        self.emb = nn.Embedding(config.TOTAL_VOCAB_SIZE, config.EMBED_DIM)

        # Standard Transformer Encoder (Post-Norm)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.EMBED_DIM,
            nhead=config.TRANSFORMER_HEADS,
            # Standard convention: feedforward dim is 4x model dim
            dim_feedforward=config.EMBED_DIM * 4,
            dropout=config.TRANSFORMER_DROPOUT,
            activation=config.TRANSFORMER_ACTIVATION,
            batch_first=True,
            norm_first=config.TRANSFORMER_NORM_FIRST,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=config.TRANSFORMER_LAYERS
        )

        # ----------------------------------------------------------------------
        # Fusion Layer
        # ----------------------------------------------------------------------
        # Stream 1 Output: Sequence Length * Embed Dim
        stream1_dim = config.SEQUENCE_LENGTH * config.EMBED_DIM
        # Stream 2 Output: Raw Continuous Features
        stream2_dim = config.NUM_CONTINUOUS_FEATURES

        fusion_input_dim = stream1_dim + stream2_dim

        # Linear Stem
        self.stem = nn.Linear(fusion_input_dim, config.BACKBONE_STAGES[0])

        # ----------------------------------------------------------------------
        # Backbone: LayerNorm SwiGLU ResFunnel
        # ----------------------------------------------------------------------
        layers = []
        in_dim = config.BACKBONE_STAGES[0]

        # Calculate Stochastic Depth schedule
        total_blocks = len(config.BACKBONE_STAGES) * config.BLOCKS_PER_STAGE
        # Linear schedule from 0.0 to STOCHASTIC_DEPTH_MAX
        dp_rates = torch.linspace(0, config.STOCHASTIC_DEPTH_MAX, total_blocks).tolist()
        block_idx = 0

        for i, out_dim in enumerate(config.BACKBONE_STAGES):
            # Transition Layer (Pre-Norm: LayerNorm -> Linear)
            # Skip for the first stage as Stem handles projection
            if i > 0:
                layers.append(nn.LayerNorm(in_dim))
                layers.append(nn.Linear(in_dim, out_dim))
                in_dim = out_dim

            # Stack Residual Blocks
            for _ in range(config.BLOCKS_PER_STAGE):
                layers.append(SwiGLUBlock(in_dim, drop_path=dp_rates[block_idx]))
                block_idx += 1

        self.backbone = nn.Sequential(*layers)

        # ----------------------------------------------------------------------
        # Head: Multi-Sample Dropout
        # ----------------------------------------------------------------------
        # 5 parallel dropout branches
        self.head_drops = nn.ModuleList(
            [nn.Dropout(config.MSD_DROPOUT_RATE) for _ in range(config.MSD_NUM_HEADS)]
        )
        # Shared Linear Output
        self.head_linear = nn.Linear(config.BACKBONE_STAGES[-1], 1)

    def forward(self, cont, cat):
        """
        Args:
            cont: Continuous features (Batch, 30)
            cat: Categorical features (Batch, 10) - Position-distinct indices
        Returns:
            logits: (Batch, MSD_NUM_HEADS)
        """
        # --- Stream 1 Processing ---
        # Embed: (B, 10) -> (B, 10, 32)
        x_cat = self.emb(cat)
        # Transform: (B, 10, 32) -> (B, 10, 32)
        x_cat = self.transformer(x_cat)
        # Flatten: (B, 10, 32) -> (B, 320)
        x_cat = x_cat.flatten(1)

        # --- Fusion ---
        # Concatenate with raw continuous features
        x = torch.cat([x_cat, cont], dim=1)

        # Stem Projection
        x = self.stem(x)

        # --- Backbone ---
        x = self.backbone(x)

        # --- Multi-Sample Dropout Head ---
        outs = []
        for drop_layer in self.head_drops:
            # Apply dropout then shared linear
            out = self.head_linear(drop_layer(x))
            outs.append(out)

        # Stack outputs: (B, 5)
        return torch.cat(outs, dim=1)
