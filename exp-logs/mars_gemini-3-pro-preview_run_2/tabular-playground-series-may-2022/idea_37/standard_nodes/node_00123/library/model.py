import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config
from library.utils import init_weights, init_transformer_weights, init_pos_embed


class SwiGLU(nn.Module):
    """
    Swish-Gated Linear Unit.
    Expects input of shape (..., 2 * dim).
    Splits into gate and value, applies SiLU to gate, and multiplies.
    Output shape: (..., dim).
    """

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return F.silu(x1) * x2


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample.
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0:
            random_tensor.div_(keep_prob)
        return x * random_tensor


class SwiGLUBlock(nn.Module):
    """
    Pre-LayerNorm Direct SwiGLU Residual Block.
    Structure: x_out = x + DropPath(Dropout(SwiGLU(Linear(LayerNorm(x)))))
    """

    def __init__(self, dim, drop_path=0.0, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        # Linear expands D -> 2D to support SwiGLU splitting
        self.linear = nn.Linear(dim, 2 * dim)
        self.act = SwiGLU()
        self.dropout = nn.Dropout(dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # Apply Kaiming init via init_weights
        self.apply(init_weights)

    def forward(self, x):
        input_x = x
        x = self.norm(x)
        x = self.linear(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.drop_path(x)
        return input_x + x


class CategoricalTransformerStream(nn.Module):
    """
    Stream 1: Categorical Sequence (Stabilized Post-Norm Transformer).
    Decomposes f_27, embeds, encodes, and normalizes the flat output.
    """

    def __init__(self):
        super().__init__()
        # Embedding with Unit Variance
        self.embedding = nn.Embedding(Config.CAT_VOCAB_SIZE, Config.EMBED_DIM)

        # Learnable Positional Embeddings with Low Variance
        self.pos_embed = nn.Parameter(
            torch.zeros(1, Config.CAT_SEQ_LEN, Config.EMBED_DIM)
        )

        # Post-Norm Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=Config.EMBED_DIM,
            nhead=Config.TRANSFORMER_HEADS,
            dim_feedforward=Config.EMBED_DIM * 4,
            dropout=Config.TRANSFORMER_DROPOUT,
            activation=Config.TRANSFORMER_ACTIVATION,
            batch_first=True,
            norm_first=Config.TRANSFORMER_NORM_FIRST,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=Config.TRANSFORMER_LAYERS
        )

        # Interface Normalization
        self.flatten_dim = Config.CAT_SEQ_LEN * Config.EMBED_DIM
        # Removed Interface Normalization to match best performing baseline (Cite solution_lesson_node_00122)

        self._init_weights()

    def _init_weights(self):
        # Apply specific initializations
        init_weights(self.embedding)  # Unit variance
        init_pos_embed(self.pos_embed)  # Low variance
        self.encoder.apply(init_transformer_weights)  # Xavier

    def forward(self, x):
        # x: (Batch, Seq_Len)
        x = self.embedding(x)  # (Batch, Seq, Dim)
        x = x + self.pos_embed
        x = self.encoder(x)
        x = x.reshape(x.size(0), -1)  # Flatten
        return x


class HybridSwiGLUNet(nn.Module):
    """
    Interface-Normalized Hybrid SwiGLU Network.
    Fuses categorical transformer stream with raw continuous features.
    """

    def __init__(self):
        super().__init__()

        # Stream 1: Categorical
        self.cat_stream = CategoricalTransformerStream()

        # Stream 2: Continuous (Identity)

        # Fusion Stem
        cat_dim = self.cat_stream.flatten_dim
        cont_dim = Config.NUM_CONT_FEATURES
        fusion_input_dim = cat_dim + cont_dim

        # Linear Stem to Backbone width
        initial_width = Config.BACKBONE_STAGES[0]
        self.stem = nn.Linear(fusion_input_dim, initial_width)

        # Backbone: SwiGLU ResFunnel
        self.stages = nn.ModuleList()
        self.transitions = nn.ModuleList()

        dims = Config.BACKBONE_STAGES
        num_stages = len(dims)
        total_blocks = num_stages * Config.BLOCKS_PER_STAGE

        block_idx = 0

        for i in range(num_stages):
            current_dim = dims[i]

            # Stack Residual Blocks
            stage_blocks = nn.ModuleList()
            for _ in range(Config.BLOCKS_PER_STAGE):
                # Linear Stochastic Depth Schedule
                dpr = Config.STOCHASTIC_DEPTH_MAX * block_idx / (total_blocks - 1)
                stage_blocks.append(
                    SwiGLUBlock(
                        dim=current_dim, drop_path=dpr, dropout=Config.MAIN_DROPOUT
                    )
                )
                block_idx += 1
            self.stages.append(nn.Sequential(*stage_blocks))

            # Transition (Pre-Norm: LayerNorm -> Linear)
            if i < num_stages - 1:
                next_dim = dims[i + 1]
                self.transitions.append(
                    nn.Sequential(
                        nn.LayerNorm(current_dim), nn.Linear(current_dim, next_dim)
                    )
                )
            else:
                self.transitions.append(nn.Identity())

        # Output Head
        final_dim = dims[-1]
        self.head = nn.Linear(final_dim, 1)

        # Initialize Backbone and Head
        self.stem.apply(init_weights)
        self.stages.apply(init_weights)
        self.transitions.apply(init_weights)
        self.head.apply(init_weights)

    def forward(self, cont_data, cat_data):
        # Stream 1
        cat_out = self.cat_stream(cat_data)

        # Stream 2
        cont_out = cont_data  # Identity

        # Fusion
        x = torch.cat([cat_out, cont_out], dim=1)
        x = self.stem(x)

        # Backbone
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i < len(self.stages) - 1:
                x = self.transitions[i](x)

        # Head
        logits = self.head(x)
        return torch.sigmoid(logits)

    def get_optimizer_params(self, weight_decay, weight_decay_bias_norm):
        """
        Separate parameters into groups for strict decoupled weight decay.
        Group 1: Linear weights, Embedding weights (decay)
        Group 2: Biases, LayerNorm, PosEmbed (no decay)
        """
        decay_params = []
        no_decay_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            # Check for no decay conditions:
            # 1. 1D parameters (biases, layernorm weights)
            # 2. Explicit bias names
            # 3. LayerNorm parameters
            # 4. Positional Embeddings
            if (
                param.ndim <= 1
                or name.endswith(".bias")
                or "norm" in name
                or "pos_embed" in name
            ):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        return [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": weight_decay_bias_norm},
        ]
