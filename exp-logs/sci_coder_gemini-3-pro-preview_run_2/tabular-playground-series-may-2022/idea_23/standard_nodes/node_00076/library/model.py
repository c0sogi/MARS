import torch
import torch.nn as nn
import math
from library import config


class PreActDirectGLUBlock(nn.Module):
    """
    Pre-Activation Direct GLU Residual Block.
    Structure: x + Dropout(GLU(Linear(BatchNorm(x))))
    """

    def __init__(self, dim, dropout):
        super().__init__()
        self.norm = nn.BatchNorm1d(dim)
        # GLU halves the dimension, so we project to 2 * dim
        self.linear = nn.Linear(dim, dim * 2)
        self.glu = nn.GLU(dim=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        x = self.linear(x)
        x = self.glu(x)
        x = self.dropout(x)
        return shortcut + x


class ResDownsample(nn.Module):
    """
    Projected Residual Connection for Downsampling between stages.
    Structure: Linear(x) + Dropout(GLU(Linear(BatchNorm(x))))
    """

    def __init__(self, dim_in, dim_out, dropout):
        super().__init__()
        self.norm = nn.BatchNorm1d(dim_in)
        self.linear_branch = nn.Linear(dim_in, dim_out * 2)
        self.glu = nn.GLU(dim=1)
        self.dropout = nn.Dropout(dropout)
        self.linear_shortcut = nn.Linear(dim_in, dim_out)

    def forward(self, x):
        # Shortcut path with projection
        shortcut = self.linear_shortcut(x)

        # Branch path
        x = self.norm(x)
        x = self.linear_branch(x)
        x = self.glu(x)
        x = self.dropout(x)

        return shortcut + x


class GELUTransformerStream(nn.Module):
    """
    Stream 1: Categorical Sequence processing using a GELU-Transformer.
    """

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(config.VOCAB_SIZE, config.EMBED_DIM)
        # Learnable Positional Embeddings
        self.pos_embedding = nn.Parameter(
            torch.randn(1, config.SEQUENCE_LENGTH, config.EMBED_DIM)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.EMBED_DIM,
            nhead=config.TRANSFORMER_HEADS,
            dim_feedforward=config.EMBED_DIM * 4,
            dropout=config.TRANSFORMER_DROPOUT,
            activation=config.TRANSFORMER_ACTIVATION,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=config.TRANSFORMER_LAYERS
        )

    def forward(self, x):
        # x: (Batch, SeqLen)
        x = self.embedding(x)  # (Batch, SeqLen, EmbedDim)
        x = x + self.pos_embedding
        x = self.transformer(x)
        x = x.flatten(start_dim=1)  # (Batch, SeqLen * EmbedDim)
        return x


class HybridNetwork(nn.Module):
    """
    Stem-Projected Sustained-Depth Hybrid Network.
    Fuses sequence and continuous data into a deep residual backbone.
    """

    def __init__(self):
        super().__init__()

        # --- Stream 1: Sequence ---
        self.transformer_stream = GELUTransformerStream()
        flat_seq_dim = config.SEQUENCE_LENGTH * config.EMBED_DIM

        # --- Fusion & Stem ---
        # Concatenate flattened sequence + continuous features
        fusion_input_dim = flat_seq_dim + config.NUM_CONTINUOUS_FEATURES
        # Dedicated Input Stem (Linear Projection)
        self.stem = nn.Linear(fusion_input_dim, config.BACKBONE_DIMS[0])

        # --- Backbone: Sustained-Depth Pre-Activation ResFunnel ---
        layers = []
        current_dim = config.BACKBONE_DIMS[0]

        # Stage 1
        for _ in range(config.BLOCKS_PER_STAGE):
            layers.append(PreActDirectGLUBlock(current_dim, config.BACKBONE_DROPOUT))

        # Subsequent Stages (Downsampling + Blocks)
        for next_dim in config.BACKBONE_DIMS[1:]:
            # Transition: Projected Residual Connection
            layers.append(ResDownsample(current_dim, next_dim, config.BACKBONE_DROPOUT))
            current_dim = next_dim

            # Sustained Depth Blocks
            for _ in range(config.BLOCKS_PER_STAGE):
                layers.append(
                    PreActDirectGLUBlock(current_dim, config.BACKBONE_DROPOUT)
                )

        self.backbone = nn.Sequential(*layers)

        # --- Output Head ---
        # Minimalist design: Single Linear layer
        self.head = nn.Linear(config.BACKBONE_DIMS[-1], 1)

        # Initialize Weights
        self._init_weights()

    def _init_weights(self):
        """
        Custom initialization logic:
        1. Kaiming Uniform for Linear layers (Backbone/Stem/Head).
        2. Xavier Uniform for Transformer Attention layers.
        3. Random Noise for Positional Embeddings.
        """
        # 1. Init Backbone, Stem, Head (Kaiming)
        # Cite Lesson 72: Avoid blanket initialization on hybrid architectures.
        for module in [self.stem, self.head, self.backbone]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                    if m.bias is not None:
                        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                        nn.init.uniform_(m.bias, -bound, bound)

        # 2. Init Transformer Stream
        # Embedding
        nn.init.normal_(self.transformer_stream.embedding.weight, mean=0, std=1.0)
        # Positional Embedding
        nn.init.normal_(self.transformer_stream.pos_embedding, mean=0, std=0.02)

        # Transformer Attention: Xavier Uniform
        for name, module in self.transformer_stream.transformer.named_modules():
            if isinstance(module, nn.MultiheadAttention):
                if module.in_proj_weight is not None:
                    nn.init.xavier_uniform_(module.in_proj_weight)
                if module.out_proj.weight is not None:
                    nn.init.xavier_uniform_(module.out_proj.weight)

    def forward(self, continuous, sequence):
        """
        Forward pass.
        Args:
            continuous: Tensor (Batch, 30)
            sequence: Tensor (Batch, 10)
        Returns:
            logits: Tensor (Batch, 1)
        """
        # Stream 1
        seq_out = self.transformer_stream(sequence)

        # Fusion (No Normalization)
        combined = torch.cat([seq_out, continuous], dim=1)

        # Stem
        x = self.stem(combined)

        # Backbone
        x = self.backbone(x)

        # Head
        logits = self.head(x)

        return logits
