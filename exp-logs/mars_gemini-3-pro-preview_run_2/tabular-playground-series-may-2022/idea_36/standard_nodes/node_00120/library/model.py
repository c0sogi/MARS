import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        # Work with any number of dimensions, assuming batch is dim 0
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output


class DirectSwiGLUBlock(nn.Module):
    """
    Pre-LayerNorm Direct SwiGLU Residual Block.
    Structure: x + DropPath(Dropout(SwiGLU(Linear(LayerNorm(x)))))

    The 'Direct' SwiGLU implies using a single Linear layer to project to 2*dim,
    splitting the output for the gate and value, and performing the element-wise
    multiplication without a final output projection matrix within the block.
    """

    def __init__(self, dim, drop_path=0.0, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        # Project to 2x dim to create Gate and Value chunks
        self.linear = nn.Linear(dim, 2 * dim)
        self.dropout = nn.Dropout(dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input_x = x

        # Pre-Norm
        x = self.norm(x)

        # Linear Projection
        x = self.linear(x)

        # SwiGLU Activation: Split -> Swish(Gate) * Value
        x1, x2 = x.chunk(2, dim=-1)
        x = F.silu(x1) * x2

        # Regularization
        x = self.dropout(x)
        x = self.drop_path(x)

        # Residual Connection
        return input_x + x


class ModalityScaledHybridSwiGLU(nn.Module):
    """
    Modality-Scaled Hybrid SwiGLU Network.
    Fuses a Post-Norm Transformer sequence stream with a continuous stream
    using learnable modality scalars, feeding into a SwiGLU ResFunnel backbone.
    """

    def __init__(self):
        super().__init__()

        # ----------------------------------------------------------------------
        # Stream 1: Categorical Sequence (Post-Norm Transformer)
        # ----------------------------------------------------------------------
        self.embedding = nn.Embedding(Config.VOCAB_SIZE, Config.EMBED_DIM)

        # Learnable Absolute Positional Embeddings
        self.pos_encoder = nn.Parameter(
            torch.randn(1, Config.SEQUENCE_LENGTH, Config.EMBED_DIM) * 0.02
        )

        # Standard Transformer Encoder (Post-Norm)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=Config.EMBED_DIM,
            nhead=Config.TRANSFORMER_HEADS,
            dim_feedforward=Config.EMBED_DIM * 4,
            dropout=Config.TRANSFORMER_DROPOUT,
            activation=Config.TRANSFORMER_ACTIVATION,
            batch_first=True,
            norm_first=Config.TRANSFORMER_NORM_FIRST,  # False
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=Config.TRANSFORMER_LAYERS
        )

        # ----------------------------------------------------------------------
        # Fusion Layer: Learnable Modality Scaling
        # ----------------------------------------------------------------------
        # Scalars to balance variance between unbounded Transformer output and normalized continuous features
        self.lambda_seq = nn.Parameter(torch.tensor(Config.INIT_SEQ_SCALAR))
        self.lambda_cont = nn.Parameter(torch.tensor(Config.INIT_CONT_SCALAR))

        # Calculate fusion dimension
        # Sequence: Flattened (Batch, Seq_Len * Embed_Dim)
        seq_flat_dim = Config.SEQUENCE_LENGTH * Config.EMBED_DIM
        fusion_dim = seq_flat_dim + Config.NUM_CONTINUOUS

        # Linear Stem
        self.stem = nn.Linear(fusion_dim, Config.BACKBONE_STAGES[0])

        # ----------------------------------------------------------------------
        # Backbone: LayerNorm SwiGLU ResFunnel
        # ----------------------------------------------------------------------
        self.backbone = nn.ModuleList()

        # Calculate stochastic depth rates linearly
        total_blocks = len(Config.BACKBONE_STAGES) * Config.BLOCKS_PER_STAGE
        dpr = [
            x.item()
            for x in torch.linspace(
                Config.STOCHASTIC_DEPTH_MIN, Config.STOCHASTIC_DEPTH_MAX, total_blocks
            )
        ]

        global_block_idx = 0

        for i, dim in enumerate(Config.BACKBONE_STAGES):
            stage = nn.ModuleList()

            # Pre-Norm Transition (LayerNorm -> Linear)
            # Required if dimensions change or strictly between stages
            # For the first stage, the stem maps to dim, so no transition needed immediately unless specified.
            # Assuming transitions are between stages (i > 0).
            if i > 0:
                prev_dim = Config.BACKBONE_STAGES[i - 1]
                transition = nn.Sequential(
                    nn.LayerNorm(prev_dim), nn.Linear(prev_dim, dim)
                )
                stage.append(transition)

            # Stacked Residual Blocks
            for _ in range(Config.BLOCKS_PER_STAGE):
                stage.append(
                    DirectSwiGLUBlock(
                        dim=dim,
                        drop_path=dpr[global_block_idx],
                        dropout=Config.BACKBONE_DROPOUT,
                    )
                )
                global_block_idx += 1

            self.backbone.append(stage)

        # ----------------------------------------------------------------------
        # Output Head
        # ----------------------------------------------------------------------
        self.head = nn.Linear(Config.BACKBONE_STAGES[-1], 1)

        # Initialize Weights
        self._init_weights()

    def _init_weights(self):
        """
        Custom weight initialization scheme:
        - Embeddings: Unit Variance (std=1.0)
        - Positional: Low Variance (std=0.02) (handled in __init__)
        - Transformer: Xavier (Glorot)
        - SwiGLU Backbone: Kaiming (He) Uniform
        """
        # 1. Embeddings
        nn.init.normal_(self.embedding.weight, mean=0.0, std=1.0)

        # 2. Iterate modules
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                # Check if module belongs to Transformer
                is_transformer = False
                # We check if the module is a submodule of transformer_encoder
                # A simple check on the name prefix is usually sufficient
                if "transformer_encoder" in name:
                    is_transformer = True

                if is_transformer:
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif "stem" in name or "head" in name or "backbone" in name:
                    # Backbone / Stem / Head -> Kaiming Uniform
                    # Using nonlinearity='relu' as proxy for Swish/SiLU in Kaiming init
                    nn.init.kaiming_uniform_(
                        m.weight, a=math.sqrt(5), nonlinearity="relu"
                    )
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, continuous, sequence):
        """
        Args:
            continuous: (Batch, 30) FloatTensor
            sequence: (Batch, 10) LongTensor
        Returns:
            logits: (Batch, 1) FloatTensor
        """
        # --- Stream 1: Sequence ---
        # Embed
        x_seq = self.embedding(sequence)  # (B, 10, 32)
        # Add Positional Encoding
        x_seq = x_seq + self.pos_encoder
        # Transformer Encoder
        x_seq = self.transformer_encoder(x_seq)
        # Flatten
        x_seq = x_seq.flatten(start_dim=1)  # (B, 320)
        # Modality Scaling
        x_seq = x_seq * self.lambda_seq

        # --- Stream 2: Continuous ---
        # Modality Scaling
        x_cont = continuous * self.lambda_cont

        # --- Fusion ---
        x = torch.cat([x_seq, x_cont], dim=1)
        x = self.stem(x)

        # --- Backbone ---
        for stage in self.backbone:
            for layer in stage:
                x = layer(x)

        # --- Output ---
        # Return logits (BCEWithLogitsLoss expected externally)
        return self.head(x)


import math
