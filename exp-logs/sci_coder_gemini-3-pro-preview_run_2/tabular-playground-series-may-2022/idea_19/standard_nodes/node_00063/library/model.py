import torch
import torch.nn as nn
from library.config import Config


class TransformerStream(nn.Module):
    """
    Stream 1: Categorical Sequence Processing with Normalized Fusion.

    Processes the decomposed character sequence through embeddings, a Transformer Encoder,
    and a final Batch Normalization step on the flattened representation to prepare
    for fusion with continuous features.
    """

    def __init__(self):
        super().__init__()
        self.embed_dim = Config.EMBED_DIM
        self.seq_len = Config.SEQ_LEN
        self.vocab_size = Config.VOCAB_SIZE

        # Token Embeddings
        self.token_embedding = nn.Embedding(self.vocab_size, self.embed_dim)

        # Learnable Positional Embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, self.seq_len, self.embed_dim))

        # Standard Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=Config.TRANSFORMER_HEADS,
            dim_feedforward=self.embed_dim * 4,
            dropout=Config.TRANSFORMER_DROPOUT,
            activation="relu",
            batch_first=True,
            norm_first=False,  # Standard Post-LN configuration
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=Config.TRANSFORMER_LAYERS
        )

        # Normalized Fusion: Flatten -> BatchNorm
        self.flatten_dim = self.seq_len * self.embed_dim
        self.bn_fusion = nn.BatchNorm1d(self.flatten_dim)

    def forward(self, x):
        # x shape: (Batch, Seq_Len)

        # Embedding + Positional Encoding
        x = self.token_embedding(x)  # (Batch, Seq, Dim)
        x = x + self.pos_embedding  # Broadcast addition

        # Transformer Processing
        x = self.transformer_encoder(x)

        # Flatten
        x = x.reshape(x.size(0), -1)

        # Normalize distribution for fusion
        x = self.bn_fusion(x)

        return x


class PreActGLUBlock(nn.Module):
    """
    Pre-Activation Direct GLU Residual Block.

    Topology: x_out = x + Dropout(GLU(Linear(BatchNorm(x))))
    """

    def __init__(self, in_features, dropout_rate):
        super().__init__()
        self.bn = nn.BatchNorm1d(in_features)
        # GLU halves the dimension, so we project to 2 * in_features
        self.linear = nn.Linear(in_features, in_features * 2)
        self.glu = nn.GLU(dim=1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        identity = x

        out = self.bn(x)
        out = self.linear(out)
        out = self.glu(out)
        out = self.dropout(out)

        return identity + out


class ResFunnelBackbone(nn.Module):
    """
    Deep Pre-Activation ResFunnel Backbone.

    Consists of multiple stages with decreasing width. Transitions between stages
    are handled by Linear projections. Inside each stage, multiple PreActGLUBlocks are stacked.
    """

    def __init__(self, input_dim):
        super().__init__()
        stages = Config.BACKBONE_STAGES  # e.g., [512, 256, 128]
        blocks_per_stage = Config.BLOCKS_PER_STAGE
        dropout = Config.BACKBONE_DROPOUT

        self.layers = nn.ModuleList()
        current_dim = input_dim

        for stage_dim in stages:
            # Transition / Projection if width changes
            if current_dim != stage_dim:
                self.layers.append(nn.Linear(current_dim, stage_dim))
                current_dim = stage_dim

            # Stack Residual Blocks
            for _ in range(blocks_per_stage):
                self.layers.append(PreActGLUBlock(current_dim, dropout))

        self.output_dim = current_dim

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class NormFusionResFunnel(nn.Module):
    """
    Normalized-Fusion Hybrid ResFunnel Network.

    Fuses a normalized transformer representation of categorical sequences with
    raw continuous features, processing them through a deep compressive backbone.
    """

    def __init__(self):
        super().__init__()

        # Stream 1: Sequence Data
        self.transformer_stream = TransformerStream()

        # Stream 2: Continuous Data (Raw features, no learnable parameters here)
        s2_dim = Config.NUM_CONT_FEATURES

        # Fusion Layer
        s1_dim = self.transformer_stream.flatten_dim
        fusion_input_dim = s1_dim + s2_dim
        backbone_start_dim = Config.BACKBONE_STAGES[0]

        self.fusion_projection = nn.Linear(fusion_input_dim, backbone_start_dim)

        # Backbone
        self.backbone = ResFunnelBackbone(backbone_start_dim)

        # Output Head
        # Simple Linear Layer (Backbone Output -> 1)
        # Note: Sigmoid is applied during inference/loss calculation, not here,
        # to ensure numerical stability with BCEWithLogitsLoss.
        self.head = nn.Linear(self.backbone.output_dim, 1)

    def forward(self, continuous, sequence):
        # Stream 1: Process Sequence
        s1 = self.transformer_stream(sequence)

        # Stream 2: Pass-through Continuous
        s2 = continuous

        # Fusion
        fused = torch.cat([s1, s2], dim=1)
        x = self.fusion_projection(fused)

        # Backbone Processing
        x = self.backbone(x)

        # Prediction
        logits = self.head(x)

        return logits
