import torch
import torch.nn as nn
from library.config import Config


class ResGatedBlock(nn.Module):
    """
    Residual Gated Block with GLU, Batch Normalization, and Dropout.
    Structure: Input -> Linear(2d) -> BN -> GLU(d) -> Dropout -> Add Input
    """

    def __init__(self, dim, dropout_rate=0.0):
        super().__init__()
        # GLU halves dimension, so project to 2*dim
        self.linear = nn.Linear(dim, dim * 2)
        self.bn = nn.BatchNorm1d(dim * 2)
        self.glu = nn.GLU(dim=1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        identity = x
        out = self.linear(x)
        out = self.bn(out)
        out = self.glu(out)
        out = self.dropout(out)
        return identity + out


class ProjectedResidualTransition(nn.Module):
    """
    Transition layer between stages with different widths.
    Uses a projected residual connection:
    Output = Activation(BN(Linear(x))) + Linear(x)
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.main_path = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU()
        )
        self.skip_path = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.main_path(x) + self.skip_path(x)


class TransformerContext(nn.Module):
    """
    Categorical stream processing f_27 tokens.
    Embeds tokens, adds learnable positional encodings, and processes via Transformer.
    """

    def __init__(
        self,
        vocab_size,
        embed_dim,
        seq_len,
        num_layers,
        num_heads,
        ff_dim,
        dropout_rate,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # Learnable positional encoding: (1, seq_len, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout_rate,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.flatten_dim = seq_len * embed_dim

    def forward(self, x):
        # x: (Batch, Seq_Len)
        x = self.embedding(x)  # (Batch, Seq_Len, Embed_Dim)
        x = x + self.pos_embedding  # Broadcast add
        x = self.transformer(x)
        # Flatten context vector
        x = x.reshape(x.size(0), -1)
        return x


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) Layer.
    Modulates signal based on context: Signal_mod = Signal * gamma(context) + beta(context)
    """

    def __init__(self, context_dim, signal_dim):
        super().__init__()
        self.gamma_proj = nn.Linear(context_dim, signal_dim)
        self.beta_proj = nn.Linear(context_dim, signal_dim)

        # Initialize for identity modulation at start (gamma=1, beta=0)
        nn.init.ones_(self.gamma_proj.weight)
        nn.init.zeros_(self.gamma_proj.bias)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)

    def forward(self, context, signal):
        gamma = self.gamma_proj(context)
        beta = self.beta_proj(context)
        return signal * gamma + beta


class FiLMResFunnel(nn.Module):
    """
    Main Architecture: Context-Modulated Hybrid ResFunnel.
    Combines a Transformer-based categorical context with continuous signals
    via FiLM modulation, feeding into a deep ResFunnel backbone.
    """

    def __init__(self):
        super().__init__()

        # ----------------------------------------------------------------------
        # Stream 1: Categorical Context
        # ----------------------------------------------------------------------
        self.context_stream = TransformerContext(
            vocab_size=Config.VOCAB_SIZE,
            embed_dim=Config.EMBED_DIM,
            seq_len=Config.SEQUENCE_LENGTH,
            num_layers=Config.TRANSFORMER_LAYERS,
            num_heads=Config.TRANSFORMER_HEADS,
            ff_dim=Config.TRANSFORMER_FF_DIM,
            dropout_rate=Config.DROPOUT_RATE,
        )
        context_out_dim = self.context_stream.flatten_dim

        # ----------------------------------------------------------------------
        # Stream 2: Continuous Signal
        # ----------------------------------------------------------------------
        self.signal_proj = nn.Sequential(
            nn.Linear(Config.NUM_CONTINUOUS_FEATURES, Config.SIGNAL_DIM),
            nn.BatchNorm1d(Config.SIGNAL_DIM),
        )

        # ----------------------------------------------------------------------
        # FiLM Fusion
        # ----------------------------------------------------------------------
        self.film = FiLMLayer(context_out_dim, Config.SIGNAL_DIM)

        # ----------------------------------------------------------------------
        # Backbone Entry Projection
        # ----------------------------------------------------------------------
        # Concatenate Context + Signal_Mod -> Project to Stage 1 width
        concat_dim = context_out_dim + Config.SIGNAL_DIM
        self.backbone_input_proj = nn.Sequential(
            nn.Linear(concat_dim, Config.RESFUNNEL_DIMS[0]),
            nn.BatchNorm1d(Config.RESFUNNEL_DIMS[0]),
            nn.ReLU(),
        )

        # ----------------------------------------------------------------------
        # ResFunnel Backbone
        # ----------------------------------------------------------------------
        layers = []
        dims = Config.RESFUNNEL_DIMS  # [512, 256, 128]
        # Using 2 blocks per stage for depth
        blocks_per_stage = 2

        for i, target_dim in enumerate(dims):
            # Transition (Downsampling) if not first stage
            if i > 0:
                prev_dim = dims[i - 1]
                layers.append(ProjectedResidualTransition(prev_dim, target_dim))

            # Blocks
            for _ in range(blocks_per_stage):
                layers.append(
                    ResGatedBlock(target_dim, dropout_rate=Config.DROPOUT_RATE)
                )

        self.backbone = nn.Sequential(*layers)

        # ----------------------------------------------------------------------
        # Output Head
        # ----------------------------------------------------------------------
        self.head = nn.Linear(dims[-1], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_cont, x_cat):
        """
        Args:
            x_cont: Continuous features (Batch, 30)
            x_cat: Categorical tokens (Batch, 10)
        """
        # 1. Process Streams
        context = self.context_stream(x_cat)
        signal = self.signal_proj(x_cont)

        # 2. FiLM Modulation
        signal_mod = self.film(context, signal)

        # 3. Concatenation & Projection
        combined = torch.cat([context, signal_mod], dim=1)
        x = self.backbone_input_proj(combined)

        # 4. Deep Backbone
        x = self.backbone(x)

        # 5. Output
        x = self.head(x)
        x = self.sigmoid(x)

        return x
