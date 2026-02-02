import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PreActGLUBlock(nn.Module):
    """
    Pre-Activation Residual Block with Gated Linear Unit (GLU).
    Structure: x -> BN -> Linear -> GLU -> Dropout -> + -> x
    Handles dimension changes via projected shortcut.
    """

    def __init__(self, in_dim, out_dim, dropout=0.35):
        super(PreActGLUBlock, self).__init__()
        self.bn = nn.BatchNorm1d(in_dim)
        # GLU requires input dimension to be 2 * target dimension
        self.linear = nn.Linear(in_dim, out_dim * 2)
        self.dropout = nn.Dropout(dropout)

        # Projected Residual Connection if dimensions change
        if in_dim != out_dim:
            self.shortcut = nn.Linear(in_dim, out_dim)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        # Pre-activation
        out = self.bn(x)
        out = self.linear(out)
        out = F.glu(out, dim=1)  # Halves the dimension
        out = self.dropout(out)

        return self.shortcut(x) + out


class TransformerStream(nn.Module):
    """
    Processes categorical sequence data using a Transformer Encoder.
    """

    def __init__(
        self,
        vocab_size=26,
        seq_len=10,
        embed_dim=32,
        nhead=4,
        num_layers=2,
        dropout=0.1,
    ):
        super(TransformerStream, self).__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim

        # Embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Learnable Positional Embeddings initialized with Random Noise
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, embed_dim))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # x: (batch, seq_len)
        x = self.embedding(x)  # (batch, seq_len, embed_dim)
        x = x + self.pos_embedding
        x = self.encoder(x)
        # Flatten: (batch, seq_len * embed_dim)
        return x.reshape(x.size(0), -1)


class HybridResFunnel(nn.Module):
    """
    Hybrid Residual Funnel Network.
    Fuses sequence and continuous data via a simple Linear Stem (Cite Lesson 78)
    and processes through a ResFunnel backbone.
    """

    def __init__(
        self,
        continuous_dim=30,
        vocab_size=30,  # 'A'-'Z' is 26, slightly higher for safety
        seq_len=10,
        embed_dim=32,
        backbone_dropout=0.35,
    ):
        super(HybridResFunnel, self).__init__()

        # --- Stream 1: Sequence ---
        self.transformer_stream = TransformerStream(
            vocab_size=vocab_size,
            seq_len=seq_len,
            embed_dim=embed_dim,
            nhead=4,
            num_layers=2,
            dropout=0.1,
        )
        transformer_out_dim = seq_len * embed_dim  # 10 * 32 = 320

        # --- Fusion: Linear Stem ---
        # Input: Flattened Transformer (320) + Raw Continuous (30) = 350
        fusion_input_dim = transformer_out_dim + continuous_dim
        stem_output_dim = 512

        # Simple Stem: Linear Projection (Cite Lesson 78, 74)
        self.stem = nn.Linear(fusion_input_dim, stem_output_dim)

        # --- Backbone: Pre-Activation ResFunnel ---
        # Stage 1: 512 -> 512
        self.stage1 = nn.Sequential(
            PreActGLUBlock(512, 512, dropout=backbone_dropout),
            PreActGLUBlock(512, 512, dropout=backbone_dropout),
        )

        # Stage 2: 512 -> 256 (Downsampling handled by first block)
        self.stage2 = nn.Sequential(
            PreActGLUBlock(512, 256, dropout=backbone_dropout),
            PreActGLUBlock(256, 256, dropout=backbone_dropout),
        )

        # Stage 3: 256 -> 128
        self.stage3 = nn.Sequential(
            PreActGLUBlock(256, 128, dropout=backbone_dropout),
            PreActGLUBlock(128, 128, dropout=backbone_dropout),
        )

        # --- Output Head ---
        self.head = nn.Linear(128, 1)

        # --- Initialization ---
        self._init_weights()

    def _init_weights(self):
        for name, m in self.named_modules():
            # Transformer Initialization (Xavier)
            if "transformer_stream" in name:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Embedding):
                    nn.init.xavier_uniform_(m.weight)

            # Backbone/Stem Initialization (Kaiming Uniform)
            else:
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                    if m.bias is not None:
                        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                        nn.init.uniform_(m.bias, -bound, bound)

    def forward(self, x_cont, x_seq):
        # Stream 1: Sequence
        seq_feat = self.transformer_stream(x_seq)

        # Stream 2: Continuous (Raw)
        cont_feat = x_cont

        # Fusion
        fused = torch.cat([seq_feat, cont_feat], dim=1)

        # Simple Stem (Cite Lesson 78)
        x = self.stem(fused)

        # Backbone
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)

        # Head
        logits = self.head(x)
        return logits
