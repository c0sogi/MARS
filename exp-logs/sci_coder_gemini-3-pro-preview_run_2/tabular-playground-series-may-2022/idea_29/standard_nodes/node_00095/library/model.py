import torch
import torch.nn as nn
import numpy as np
from library.config import Config
from library.layers import PreNormResBlock, TransitionLayer


class LateFusionSwishGatedResFunnel(nn.Module):
    def __init__(self):
        super().__init__()

        # --- Stream 1: Categorical Sequence ---
        # 26 characters + 1 for potential unknown/padding
        self.char_embed = nn.Embedding(27, Config.EMBED_DIM)

        # Removed State Token Projection (Cite solution_lesson_node_00093)

        # Positional Embedding: 10 chars
        # Initialized with low variance noise N(0, 0.02) (Cite solution_lesson_node_00081)
        self.pos_embed = nn.Parameter(
            torch.randn(1, Config.SEQ_LEN, Config.EMBED_DIM) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=Config.EMBED_DIM,
            nhead=Config.TRANSFORMER_HEADS,
            dim_feedforward=Config.EMBED_DIM * 4,
            dropout=Config.TRANSFORMER_DROPOUT,
            activation="gelu",  # Cite solution_lesson_node_00068
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=Config.TRANSFORMER_LAYERS
        )

        # Flattened output dim: 10 chars * 32
        self.flat_dim = Config.SEQ_LEN * Config.EMBED_DIM

        # --- Fusion ---
        # Input to fusion: Flattened Transformer (320) + Raw Continuous (30)
        # Late Fusion strategy (Cite solution_lesson_node_00093)
        fusion_input_dim = self.flat_dim + 30
        self.stem = nn.Linear(fusion_input_dim, Config.BACKBONE_STAGES[0])

        # --- Backbone: Swish-Gated ResFunnel ---
        self.stages = nn.ModuleList()
        self.transitions = nn.ModuleList()

        # Stochastic depth schedule
        total_blocks = len(Config.BACKBONE_STAGES) * Config.BLOCKS_PER_STAGE
        dp_rates = torch.linspace(0, Config.STOCHASTIC_DEPTH_MAX, total_blocks)

        block_idx = 0
        for i, stage_dim in enumerate(Config.BACKBONE_STAGES):
            # Transition (Downsample/Project) if not first stage
            if i > 0:
                prev_dim = Config.BACKBONE_STAGES[i - 1]
                self.transitions.append(TransitionLayer(prev_dim, stage_dim))
            else:
                self.transitions.append(nn.Identity())

            # Blocks
            stage_blocks = nn.ModuleList()
            for _ in range(Config.BLOCKS_PER_STAGE):
                stage_blocks.append(
                    PreNormResBlock(
                        dim=stage_dim,
                        dropout=Config.BACKBONE_DROPOUT,
                        drop_path=dp_rates[block_idx].item(),
                    )
                )
                block_idx += 1
            self.stages.append(stage_blocks)

        # --- Head ---
        self.final_norm = nn.LayerNorm(Config.BACKBONE_STAGES[-1])
        self.head = nn.Linear(Config.BACKBONE_STAGES[-1], 1)

        self._init_weights()

    def _init_weights(self):
        # General Initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=np.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # Transformer Specific Initialization (Xavier)
        # This overrides the Kaiming init applied above for transformer parameters
        for p in self.transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x_cont, x_cat):
        # x_cont: (B, 30)
        # x_cat: (B, 10)

        B = x_cont.shape[0]

        # --- Stream 1: Categorical Sequence ---
        # Embed chars
        seq = self.char_embed(x_cat)  # (B, 10, 32)

        # Add Positional Embeddings
        seq = seq + self.pos_embed

        # Transformer
        seq_out = self.transformer(seq)  # (B, 10, 32)

        # Flatten
        flat_chars = seq_out.reshape(B, -1)  # (B, 320)

        # --- Fusion ---
        # Late Fusion: Concat Flattened Sequence with raw continuous features
        # Cite solution_lesson_node_00093
        fused = torch.cat([flat_chars, x_cont], dim=1)  # (B, 350)

        # Linear Stem (Cite solution_lesson_node_00074)
        x = self.stem(fused)

        # --- Backbone ---
        for i, stage in enumerate(self.stages):
            # Transition
            if not isinstance(self.transitions[i], nn.Identity):
                x = self.transitions[i](x)

            # Blocks
            for block in stage:
                x = block(x)

        x = self.final_norm(x)
        logits = self.head(x)
        return logits
