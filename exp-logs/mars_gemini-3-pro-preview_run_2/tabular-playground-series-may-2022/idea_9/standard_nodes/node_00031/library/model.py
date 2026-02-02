import torch
import torch.nn as nn
from library.config import Config


class GLUBlock(nn.Module):
    """
    Gated Linear Unit Block with Residual Connection.
    Structure: x + Dropout(BatchNorm(GLU(Linear(x))))
    Handles dimension changes via Projected Residual Connection on the skip path.
    """

    def __init__(self, in_dim, out_dim, dropout_rate=0.0):
        super(GLUBlock, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Main path
        # GLU halves the dimension, so we project to out_dim * 2
        self.linear = nn.Linear(in_dim, out_dim * 2)
        self.glu = nn.GLU(dim=1)
        self.bn = nn.BatchNorm1d(out_dim)
        self.dropout = nn.Dropout(dropout_rate)

        # Skip path (Projected Residual Connection)
        if in_dim != out_dim:
            self.skip_proj = nn.Linear(in_dim, out_dim)
        else:
            self.skip_proj = nn.Identity()

    def forward(self, x):
        # Skip connection
        skip = self.skip_proj(x)

        # Main path
        out = self.linear(x)
        out = self.glu(out)
        out = self.bn(out)
        out = self.dropout(out)

        # Residual add
        return skip + out


class HybridAttentionResFunnel(nn.Module):
    """
    Hybrid architecture combining Sequence-Aware Categorical processing (Transformer)
    with a Deep Residual Funnel Gated Network (ResFunnel-GLU).
    """

    def __init__(self):
        super(HybridAttentionResFunnel, self).__init__()

        # ----------------------------------------------------------------------
        # Stream 1: Sequence-Aware Categorical Processing
        # ----------------------------------------------------------------------
        self.embedding = nn.Embedding(Config.VOCAB_SIZE, Config.EMBED_DIM)

        # Transformer Encoder Layer
        # Captures dependencies between characters in the f_27 string
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=Config.EMBED_DIM,
            nhead=Config.TRANSFORMER_HEADS,
            dim_feedforward=Config.TRANSFORMER_FF_DIM,
            dropout=Config.DROPOUT_RATE,
            batch_first=True,
            norm_first=False,  # Standard Post-LN as per description
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=Config.TRANSFORMER_LAYERS
        )

        # Flatten dimension: Sequence Length * Embedding Dim
        self.cat_flat_dim = Config.F_27_SEQ_LEN * Config.EMBED_DIM

        # ----------------------------------------------------------------------
        # Stream 2: Continuous Processing
        # ----------------------------------------------------------------------
        self.cont_bn = nn.BatchNorm1d(Config.NUM_CONT_FEATURES)

        # ----------------------------------------------------------------------
        # Backbone: Residual Funnel Gated Network
        # ----------------------------------------------------------------------
        # Fusion Dimension
        fusion_dim = self.cat_flat_dim + Config.NUM_CONT_FEATURES

        # Initial Projection
        initial_width = Config.HIDDEN_SIZES[0]
        self.input_proj = nn.Linear(fusion_dim, initial_width)

        # Construct Stages
        # Config.HIDDEN_SIZES = [512, 256, 128]
        layers = []
        current_dim = initial_width

        # We construct the funnel. For each target width, we add blocks.
        # We ensure at least one block performs the transition (downsampling) if needed,
        # followed by processing blocks.

        # Stage 1: Width 512
        # Since input is already projected to 512, we just add processing blocks
        layers.append(GLUBlock(current_dim, 512, Config.DROPOUT_RATE))
        layers.append(GLUBlock(512, 512, Config.DROPOUT_RATE))
        current_dim = 512

        # Stage 2: Width 256
        # Transition block (512 -> 256)
        layers.append(GLUBlock(current_dim, 256, Config.DROPOUT_RATE))
        # Processing block
        layers.append(GLUBlock(256, 256, Config.DROPOUT_RATE))
        current_dim = 256

        # Stage 3: Width 128
        # Transition block (256 -> 128)
        layers.append(GLUBlock(current_dim, 128, Config.DROPOUT_RATE))
        # Processing block
        layers.append(GLUBlock(128, 128, Config.DROPOUT_RATE))
        current_dim = 128

        self.backbone = nn.Sequential(*layers)

        # ----------------------------------------------------------------------
        # Output Head
        # ----------------------------------------------------------------------
        self.head = nn.Linear(current_dim, 1)

    def forward(self, cont_features, cat_sequence):
        """
        Args:
            cont_features: Tensor (Batch, 30)
            cat_sequence: Tensor (Batch, 10) - Integer encoded
        Returns:
            logits: Tensor (Batch, 1)
        """
        # --- Stream 1 ---
        # Embed: (B, 10) -> (B, 10, 32)
        emb = self.embedding(cat_sequence)
        # Transform: (B, 10, 32) -> (B, 10, 32)
        trans_out = self.transformer(emb)
        # Flatten: (B, 10, 32) -> (B, 320)
        cat_flat = trans_out.reshape(trans_out.size(0), -1)

        # --- Stream 2 ---
        # Normalize continuous features
        cont_norm = self.cont_bn(cont_features)

        # --- Fusion ---
        x = torch.cat([cat_flat, cont_norm], dim=1)
        x = self.input_proj(x)

        # --- Backbone ---
        x = self.backbone(x)

        # --- Head ---
        # Returning logits for numerical stability with BCEWithLogitsLoss
        logits = self.head(x)

        return logits
