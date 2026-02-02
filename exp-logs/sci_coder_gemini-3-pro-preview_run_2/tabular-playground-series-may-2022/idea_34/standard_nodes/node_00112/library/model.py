import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class SwiGLU(nn.Module):
    """
    Swish-Gated Linear Unit activation.
    Takes an input of dimension 2*D, splits it into (x, gate),
    and returns output of dimension D: x * SiLU(gate).
    """

    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * F.silu(gate)


class DropPath(nn.Module):
    """
    Stochastic Depth (DropPath) regularization.
    Randomly drops residual paths during training.
    """

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        # Support arbitrary dimensions, assuming batch is dim 0
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0:
            random_tensor.div_(keep_prob)
        return x * random_tensor


class ResidualBlock(nn.Module):
    """
    Pre-LayerNorm Direct SwiGLU Residual Block.
    Structure: x_out = x + DropPath(Dropout(SwiGLU(Linear(LayerNorm(x)))))
    """

    def __init__(self, dim, drop_path=0.0, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        # Linear maps dim -> 2*dim to provide inputs for SwiGLU splitting
        self.linear = nn.Linear(dim, 2 * dim)
        self.swiglu = SwiGLU()
        self.dropout = nn.Dropout(dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input_x = x
        x = self.norm(x)
        x = self.linear(x)
        x = self.swiglu(x)
        x = self.dropout(x)
        x = self.drop_path(x)
        return input_x + x


class DualViewResFunnel(nn.Module):
    """
    Dual-View Post-Norm SwiGLU-ResFunnel Network.
    Fuses a dual-view categorical representation (Transformer + Linear Shortcut)
    with continuous features into a deep SwiGLU-based residual funnel.
    """

    def __init__(
        self,
        vocab_size=Config.VOCAB_SIZE,
        embedding_dim=Config.EMBEDDING_DIM,
        seq_len=Config.SEQUENCE_LENGTH,
        num_cont=Config.NUM_CONT_FEATURES,
        transformer_layers=Config.TRANSFORMER_LAYERS,
        transformer_heads=Config.TRANSFORMER_HEADS,
        transformer_dropout=Config.TRANSFORMER_DROPOUT,
        backbone_stages=Config.BACKBONE_STAGES,
        backbone_blocks=Config.BACKBONE_BLOCKS,
        backbone_dropout=Config.BACKBONE_DROPOUT,
        stochastic_depth_max=Config.STOCHASTIC_DEPTH_MAX,
    ):
        super().__init__()

        # ----------------------------------------------------------------------
        # 1. Embeddings & Positional Encoding
        # ----------------------------------------------------------------------
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # Explicitly initialize with Low Variance Random Noise
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, embedding_dim))

        # ----------------------------------------------------------------------
        # 2. View A: Contextual (Transformer Encoder)
        # ----------------------------------------------------------------------
        # Post-Normalization (norm_first=False), GELU activation
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=transformer_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=transformer_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=transformer_layers
        )

        # ----------------------------------------------------------------------
        # 3. Fusion Stem
        # ----------------------------------------------------------------------
        # View A (Transformer) -> Flattened: seq_len * embedding_dim
        # View B (Linear Shortcut) -> Flattened: seq_len * embedding_dim
        # Continuous Features -> num_cont
        fusion_input_dim = (seq_len * embedding_dim) * 2 + num_cont
        stem_output_dim = backbone_stages[0]

        # Linear Stem (No normalization immediately after concat)
        self.stem = nn.Linear(fusion_input_dim, stem_output_dim)

        # ----------------------------------------------------------------------
        # 4. Backbone: LayerNorm SwiGLU ResFunnel
        # ----------------------------------------------------------------------
        self.stages = nn.ModuleList()
        self.transitions = nn.ModuleList()

        total_blocks = sum(backbone_blocks)
        global_block_idx = 0

        for i, (dim, num_blocks) in enumerate(zip(backbone_stages, backbone_blocks)):
            # Build blocks for this stage
            stage_blocks = []
            for _ in range(num_blocks):
                # Linear Stochastic Depth schedule
                sd_rate = stochastic_depth_max * global_block_idx / (total_blocks - 1)
                stage_blocks.append(
                    ResidualBlock(dim, drop_path=sd_rate, dropout=backbone_dropout)
                )
                global_block_idx += 1
            self.stages.append(nn.Sequential(*stage_blocks))

            # Build transition to next stage (if not last stage)
            if i < len(backbone_stages) - 1:
                next_dim = backbone_stages[i + 1]
                # Pre-Norm Transition: LayerNorm -> Linear
                self.transitions.append(
                    nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, next_dim))
                )
            else:
                self.transitions.append(nn.Identity())

        # ----------------------------------------------------------------------
        # 5. Output Head
        # ----------------------------------------------------------------------
        final_dim = backbone_stages[-1]
        self.head = nn.Linear(final_dim, 1)

        # Apply specific weight initialization
        self._init_weights()

    def _init_weights(self):
        """
        Applies specific initialization schemes:
        - Positional Embeddings: Normal(0, 0.02)
        - Transformer: Xavier (Glorot) Uniform
        - SwiGLU Blocks: Kaiming (He) Uniform
        """
        # Positional Embeddings
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)

        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if "transformer" in name:
                    # Transformer: Xavier
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif "stages" in name:
                    # Backbone (SwiGLU Blocks): Kaiming Uniform
                    # a=sqrt(5) is the default for Linear in PyTorch, but explicit Kaiming is safer for Swish/ReLU
                    nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                else:
                    # Stem, Transitions, Head: Default Init (usually Kaiming Uniform)
                    pass
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding) and "embedding" in name:
                # Standard embedding init (optional, usually N(0,1) or U(-1,1))
                nn.init.normal_(m.weight, mean=0.0, std=1.0)

    def forward(self, cat_data, cont_data):
        """
        Args:
            cat_data: (Batch, 10) LongTensor - Categorical character indices
            cont_data: (Batch, 30) FloatTensor - Normalized continuous features
        Returns:
            (Batch, 1) FloatTensor - Predicted probabilities
        """
        B = cat_data.shape[0]

        # 1. Embeddings
        x_emb = self.embedding(cat_data)  # (B, 10, 32)
        x_emb = x_emb + self.pos_embed  # Add learnable positional encoding

        # 2. View A: Contextual (Transformer)
        x_trans = self.transformer(x_emb)  # (B, 10, 32)
        x_view_a = x_trans.flatten(1)  # (B, 320)

        # 3. View B: Local (Linear Shortcut)
        x_view_b = x_emb.flatten(1)  # (B, 320)

        # 4. Fusion
        x_concat = torch.cat([x_view_a, x_view_b, cont_data], dim=1)  # (B, 670)
        x = self.stem(x_concat)  # (B, 512)

        # 5. Backbone
        for stage, transition in zip(self.stages, self.transitions):
            x = stage(x)
            x = transition(x)

        # 6. Head
        logits = self.head(x)
        return torch.sigmoid(logits)
