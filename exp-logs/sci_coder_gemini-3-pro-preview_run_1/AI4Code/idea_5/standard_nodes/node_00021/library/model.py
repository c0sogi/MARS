import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from library.config import Config


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(SinusoidalPositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        # x: (Batch, Seq_Len, Dim)
        # Returns: (Batch, Seq_Len, Dim)
        return x + self.pe[:, : x.size(1), :]


class ProjectionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super(ProjectionHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class DCAN(nn.Module):
    def __init__(self):
        super(DCAN, self).__init__()

        # Hyperparameters
        input_dim = Config.INPUT_DIM
        hidden_dim = Config.HIDDEN_DIM
        nhead = Config.NHEAD
        num_layers = Config.NUM_ENCODER_LAYERS
        dim_feedforward = Config.DIM_FEEDFORWARD
        dropout = Config.DROPOUT

        # 1. Projections
        self.code_projector = ProjectionHead(input_dim, hidden_dim, dropout)
        self.md_projector = ProjectionHead(input_dim, hidden_dim, dropout)

        # 2. Code Branch Components
        self.pos_encoder = SinusoidalPositionalEncoding(hidden_dim, max_len=1000)
        # Learnable Sink Token representing the position after the last code cell
        self.sink_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        code_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.code_encoder = nn.TransformerEncoder(code_layer, num_layers=num_layers)

        # 3. Markdown Branch Components (Set Transformer - No Positional Encoding)
        md_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.md_encoder = nn.TransformerEncoder(md_layer, num_layers=num_layers)

        # Initialization
        self._init_weights()

    def _init_weights(self):
        # Initialize sink token and linear layers
        nn.init.normal_(self.sink_token, mean=0, std=0.02)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, code_features, code_mask, markdown_features, markdown_mask):
        """
        Args:
            code_features: (B, L, Input_Dim)
            code_mask: (B, L) - True for valid tokens, False for padding
            markdown_features: (B, M, Input_Dim)
            markdown_mask: (B, M) - True for valid tokens, False for padding

        Returns:
            logits: (B, M, L+1) - Scores for placing MD cells before Code[0]...Code[L-1] or at End (Sink)
        """
        batch_size = code_features.size(0)

        # =================================================
        # 1. Process Code Sequence (Anchors)
        # =================================================
        # Project
        code_emb = self.code_projector(code_features)  # (B, L, H)

        # Add Positional Encoding
        code_emb = self.pos_encoder(code_emb)

        # Append Sink Token
        # Expand sink token to batch size: (B, 1, H)
        sink_emb = self.sink_token.expand(batch_size, -1, -1)

        # Concatenate: (B, L+1, H)
        code_full = torch.cat([code_emb, sink_emb], dim=1)

        # Update Code Mask
        # The sink token is always valid (True)
        sink_mask = torch.ones(
            (batch_size, 1), device=code_mask.device, dtype=torch.bool
        )
        code_full_mask = torch.cat([code_mask, sink_mask], dim=1)  # (B, L+1)

        # Transformer Encoder
        # PyTorch expects key_padding_mask where True = Padding (Ignore)
        # Our mask is True = Valid. So we invert it.
        code_ctx = self.code_encoder(
            code_full, src_key_padding_mask=~code_full_mask
        )  # (B, L+1, H)

        # =================================================
        # 2. Process Markdown Set (Queries)
        # =================================================
        # Project
        md_emb = self.md_projector(markdown_features)  # (B, M, H)

        # No Positional Encoding (Set property)

        # Transformer Encoder
        md_ctx = self.md_encoder(
            md_emb, src_key_padding_mask=~markdown_mask
        )  # (B, M, H)

        # =================================================
        # 3. Interaction Head (Dot Product Attention)
        # =================================================
        # Query: MD Context (B, M, H)
        # Key: Code Context (B, L+1, H)
        # Logits: (B, M, L+1)

        # Normalize for scaled dot product
        scale = math.sqrt(Config.HIDDEN_DIM)
        logits = torch.matmul(md_ctx, code_ctx.transpose(1, 2)) / scale

        # =================================================
        # 4. Masking Output Logits
        # =================================================
        # We must mask positions in the code sequence that are padding.
        # code_full_mask is (B, L+1). We need to broadcast to (B, M, L+1).

        # Expand mask: (B, 1, L+1)
        logit_mask = code_full_mask.unsqueeze(1)

        # Fill padding positions with -inf
        logits = logits.masked_fill(~logit_mask, -1e9)

        return logits
