import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from library.config import Config, SwiGLU, DropPath
from library.utils import init_weights


class SwiGLUBlock(nn.Module):
    """
    Pre-LayerNorm Direct SwiGLU Residual Block.
    Structure: x + DropPath(Dropout(SwiGLU(LayerNorm(x))))
    Cite Lesson 00087: Aggressive regularization (Dropout) inside GLU blocks prevents overfitting.
    """

    def __init__(self, dim, drop_path=0.0, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.swiglu = SwiGLU(dim)
        self.drop_path = DropPath(drop_path)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.drop_path(self.dropout(self.swiglu(self.norm(x))))


class TransformerStream(nn.Module):
    """
    Stream 1: Categorical Sequence processing using a Post-Norm Transformer.
    """

    def __init__(self, vocab_size, embed_dim, max_len, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # Learnable Absolute Positional Embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, embed_dim))

        # Post-Norm Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=4,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,  # Post-Normalization
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, x):
        # x: [Batch, Seq_Len]
        x = self.embedding(x)
        x = x + self.pos_embed
        x = self.encoder(x)
        return x.flatten(1)  # [Batch, Seq_Len * Embed_Dim]


class MultiSampleDropoutHead(nn.Module):
    """
    Output Head with Multi-Sample Dropout for better generalization.
    """

    def __init__(self, in_dim, num_heads, dropout_prob):
        super().__init__()
        self.num_heads = num_heads
        self.drop = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, x):
        if self.training:
            # Return stacked logits for multi-sample loss calculation
            logits = []
            for _ in range(self.num_heads):
                logits.append(self.fc(self.drop(x)))
            return torch.stack(logits, dim=0).squeeze(-1)  # [Heads, Batch]
        else:
            # Inference: Single pass without dropout (equivalent to averaging ensemble)
            return self.fc(x).squeeze(-1)


class PostNormHybridSwiGLU(nn.Module):
    """
    Post-Norm Hybrid SwiGLU Network with Multi-Sample Dropout.
    Fuses a sequence transformer stream with raw continuous features.
    """

    def __init__(self, config=Config):
        super().__init__()

        # Stream 1: Categorical Sequence
        self.transformer = TransformerStream(
            vocab_size=config.VOCAB_SIZE,
            embed_dim=config.EMBED_DIM,
            max_len=config.SEQ_LEN,
            dropout=config.DROPOUT_TRANSFORMER,
        )
        seq_out_dim = config.SEQ_LEN * config.EMBED_DIM

        # Stream 2: Continuous Features (Raw)
        cont_dim = config.NUM_FEATURES

        # Fusion Layer: Linear Stem
        fusion_dim = seq_out_dim + cont_dim
        self.stem = nn.Linear(fusion_dim, config.BACKBONE_STAGES[0])

        # Backbone: LayerNorm SwiGLU ResFunnel
        layers = []
        in_dim = config.BACKBONE_STAGES[0]

        # Calculate Stochastic Depth rates linearly
        total_blocks = len(config.BACKBONE_STAGES) * config.BLOCKS_PER_STAGE
        dp_rates = torch.linspace(0, config.STOCHASTIC_DEPTH_MAX, total_blocks).tolist()
        block_cnt = 0

        for stage_dim in config.BACKBONE_STAGES:
            # Pre-Norm Transition if dimension changes
            if in_dim != stage_dim:
                layers.append(nn.LayerNorm(in_dim))
                layers.append(nn.Linear(in_dim, stage_dim))
                in_dim = stage_dim

            # Stacked Residual Blocks
            for _ in range(config.BLOCKS_PER_STAGE):
                layers.append(
                    SwiGLUBlock(
                        stage_dim,
                        drop_path=dp_rates[block_cnt],
                        dropout=config.DROPOUT_BLOCK,
                    )
                )
                block_cnt += 1

        self.backbone = nn.Sequential(*layers)
        self.backbone_dropout = nn.Dropout(config.DROPOUT_BACKBONE)

        # Output Head
        self.head = MultiSampleDropoutHead(
            in_dim=config.BACKBONE_STAGES[-1],
            num_heads=config.MSD_HEADS,
            dropout_prob=config.DROPOUT_HEAD,
        )

        # Initialization
        self.apply(init_weights)

        # Override initialization for Transformer Linear layers to use Xavier (Glorot)
        # init_weights applies Kaiming to all Linears, so we correct this for the Transformer
        def init_transformer_linear(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.transformer.encoder.apply(init_transformer_linear)

    def forward(self, seq, cont):
        # Stream 1
        seq_feat = self.transformer(seq)

        # Fusion (Concatenate then Stem)
        x = torch.cat([seq_feat, cont], dim=1)
        x = self.stem(x)

        # Backbone
        x = self.backbone(x)
        x = self.backbone_dropout(x)

        # Head
        return self.head(x)
