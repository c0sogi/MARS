import torch
import torch.nn as nn
from library.config import Config
from library.layers import LayerScaledBlock
from library.utils import init_weights


class HybridSwiGLUNet(nn.Module):
    """
    LayerScaled-Fusion Hybrid SwiGLU Network.

    Fuses a stabilized Post-Norm Transformer categorical stream with a raw continuous
    stream into a deep, LayerScaled residual funnel backbone.
    """

    def __init__(self):
        super().__init__()

        # ----------------------------------------------------------------------
        # Stream 1: Categorical Sequence (Stabilized Post-Norm Transformer)
        # ----------------------------------------------------------------------
        self.embedding = nn.Embedding(Config.VOCAB_SIZE, Config.EMBED_DIM)

        # Learnable Absolute Positional Embeddings
        # Shape: (1, Sequence Length, Embedding Dim) for broadcasting
        self.pos_embedding = nn.Parameter(
            torch.randn(1, Config.SEQUENCE_LENGTH, Config.EMBED_DIM)
        )

        # Transformer Encoder
        # Configured as Post-Norm (norm_first=False) with GELU activation
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=Config.EMBED_DIM,
            nhead=Config.TRANSFORMER_HEADS,
            dim_feedforward=Config.EMBED_DIM * 4,
            dropout=Config.TRANSFORMER_DROPOUT,
            activation=Config.TRANSFORMER_ACTIVATION,
            batch_first=True,
            norm_first=False,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=Config.TRANSFORMER_LAYERS
        )

        # Branch Normalization
        # Normalizes the flattened sequence before fusion
        self.branch_norm = nn.LayerNorm(Config.SEQUENCE_LENGTH * Config.EMBED_DIM)

        # ----------------------------------------------------------------------
        # Stream 2: Continuous Preservation
        # ----------------------------------------------------------------------
        # No specific layers; features are passed raw (normalized)

        # ----------------------------------------------------------------------
        # Fusion Layer: The Linear Stem
        # ----------------------------------------------------------------------
        fusion_input_dim = (
            Config.SEQUENCE_LENGTH * Config.EMBED_DIM
        ) + Config.NUM_CONTINUOUS_FEATURES
        self.stem = nn.Linear(fusion_input_dim, Config.BACKBONE_STAGES[0])

        # ----------------------------------------------------------------------
        # Backbone: LayerScaled SwiGLU ResFunnel
        # ----------------------------------------------------------------------
        self.stages = nn.ModuleList()
        self.transitions = nn.ModuleList()

        # Calculate Stochastic Depth rates (Linear decay from 0.0 to Max)
        total_blocks = sum([Config.BLOCKS_PER_STAGE] * len(Config.BACKBONE_STAGES))
        dpr = [
            x.item()
            for x in torch.linspace(0, Config.STOCHASTIC_DEPTH_MAX_RATE, total_blocks)
        ]
        block_idx = 0

        # Construct Stages
        for i, stage_dim in enumerate(Config.BACKBONE_STAGES):
            # 1. Build Blocks for this stage
            blocks = []
            for _ in range(Config.BLOCKS_PER_STAGE):
                blocks.append(
                    LayerScaledBlock(
                        dim=stage_dim,
                        drop_path=dpr[block_idx],
                        layer_scale_init=Config.LAYERSCALE_INIT,
                        dropout=Config.BACKBONE_DROPOUT,
                    )
                )
                block_idx += 1
            self.stages.append(nn.Sequential(*blocks))

            # 2. Build Transition to next stage (if not the last stage)
            if i < len(Config.BACKBONE_STAGES) - 1:
                next_dim = Config.BACKBONE_STAGES[i + 1]
                # Pre-Norm Transition: LayerNorm -> Linear
                self.transitions.append(
                    nn.Sequential(
                        nn.LayerNorm(stage_dim), nn.Linear(stage_dim, next_dim)
                    )
                )

        # ----------------------------------------------------------------------
        # Output Head
        # ----------------------------------------------------------------------
        self.head = nn.Linear(Config.BACKBONE_STAGES[-1], 1)

        # ----------------------------------------------------------------------
        # Initialization
        # ----------------------------------------------------------------------
        # Apply standard initialization protocols defined in library.utils
        self.apply(init_weights)

        # Apply specific initialization for Positional Embeddings
        # "Low Variance Random Noise (std=0.02)"
        nn.init.normal_(self.pos_embedding, mean=0.0, std=0.02)

    def forward(self, continuous, categorical):
        """
        Args:
            continuous: Tensor of shape (Batch, 30)
            categorical: Tensor of shape (Batch, 10)
        Returns:
            logits: Tensor of shape (Batch, 1)
        """
        # --- Stream 1: Categorical ---
        # Embed: (B, 10) -> (B, 10, 32)
        x_cat = self.embedding(categorical)

        # Add Positional Encoding (Broadcasting)
        x_cat = x_cat + self.pos_embedding

        # Transformer Encoder
        x_cat = self.transformer_encoder(x_cat)

        # Flatten: (B, 10, 32) -> (B, 320)
        x_cat = x_cat.flatten(1)

        # Branch Norm
        x_cat = self.branch_norm(x_cat)

        # --- Stream 2: Continuous ---
        x_cont = continuous

        # --- Fusion ---
        # Concatenate: (B, 320) + (B, 30) -> (B, 350)
        x = torch.cat([x_cat, x_cont], dim=1)

        # Linear Stem -> (B, 512)
        x = self.stem(x)

        # --- Backbone ---
        # Stage 1 (512)
        x = self.stages[0](x)
        x = self.transitions[0](x)  # 512 -> 256

        # Stage 2 (256)
        x = self.stages[1](x)
        x = self.transitions[1](x)  # 256 -> 128

        # Stage 3 (128)
        x = self.stages[2](x)

        # --- Head ---
        logits = self.head(x)

        return logits
