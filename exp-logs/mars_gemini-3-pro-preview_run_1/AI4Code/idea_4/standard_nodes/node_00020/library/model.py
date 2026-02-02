import torch
import torch.nn as nn
import math
from library.config import Config


class PositionalEncoding(nn.Module):
    """
    Standard Sinusoidal Positional Encoding module.
    Injects information about the relative or absolute position of the tokens in the sequence.
    """

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
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
        # Returns x + positional_encoding
        return x + self.pe[:, : x.size(1)]


class DCCodeBERT(nn.Module):
    """
    Dual-Context CodeBERT Network (DC-CodeBERT).

    This model aligns shuffled markdown cells (Queries) against an ordered sequence of
    code cells (Anchors).

    Key Components:
    1. Symmetric Projection: Maps CodeBERT embeddings to a shared latent space.
    2. Code Encoder: Transformer with Positional Encoding (captures execution flow).
    3. Markdown Encoder: Transformer WITHOUT Positional Encoding (captures set context).
    4. Cross-Attention Head: Predicts the index of the code cell preceding the markdown cell.
    """

    def __init__(self):
        super(DCCodeBERT, self).__init__()

        self.hidden_dim = Config.HIDDEN_DIM
        self.latent_dim = Config.LATENT_DIM
        self.dropout_prob = Config.DROPOUT

        # ==========================================
        # 1. Projection Layers
        # ==========================================
        # Project 768-dim CodeBERT embeddings to 512-dim latent space
        self.code_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.latent_dim),
            nn.LayerNorm(self.latent_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_prob),
        )

        self.md_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.latent_dim),
            nn.LayerNorm(self.latent_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_prob),
        )

        # ==========================================
        # 2. Special Tokens
        # ==========================================
        # Learnable EOS token to represent the position after the last code cell
        # (or the end of the notebook).
        self.eos_token = nn.Parameter(torch.randn(1, 1, self.latent_dim))

        # ==========================================
        # 3. Context Encoders
        # ==========================================

        # Code Branch: Sequence Context
        # We add positional encodings because code order matters (execution flow).
        # Max len + buffer for EOS
        self.pos_encoder = PositionalEncoding(
            self.latent_dim, max_len=Config.MAX_LEN + 16
        )

        encoder_layer_code = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=8,
            dim_feedforward=2048,
            dropout=self.dropout_prob,
            batch_first=True,
        )
        self.code_encoder = nn.TransformerEncoder(encoder_layer_code, num_layers=2)

        # Markdown Branch: Set Context
        # We DO NOT add positional encodings because the input is a shuffled set.
        # The model must rely on content semantics (Set Transformer).
        encoder_layer_md = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=8,
            dim_feedforward=2048,
            dropout=self.dropout_prob,
            batch_first=True,
        )
        self.md_encoder = nn.TransformerEncoder(encoder_layer_md, num_layers=2)

        # Scale factor for dot-product attention
        self.scale = math.sqrt(self.latent_dim)

    def forward(self, code_embeddings, code_mask, md_embeddings, md_mask):
        """
        Forward pass.

        Args:
            code_embeddings (Tensor): (Batch, L_code, Hidden)
            code_mask (Tensor): (Batch, L_code) - 1 for valid, 0 for pad.
            md_embeddings (Tensor): (Batch, L_md, Hidden)
            md_mask (Tensor): (Batch, L_md) - 1 for valid, 0 for pad.

        Returns:
            logits (Tensor): (Batch, L_md, L_code + 1)
                             Scores for placing each markdown cell after each code cell
                             (including the EOS position).
        """
        batch_size = code_embeddings.size(0)

        # 1. Projection
        code_h = self.code_proj(code_embeddings)  # (B, Lc, D)
        md_h = self.md_proj(md_embeddings)  # (B, Lm, D)

        # 2. Prepare Code Sequence (Anchors)
        # Append EOS token to the sequence
        eos = self.eos_token.expand(batch_size, -1, -1)  # (B, 1, D)
        code_h = torch.cat([code_h, eos], dim=1)  # (B, Lc+1, D)

        # Update code mask to include EOS (always valid)
        eos_mask = torch.ones(
            (batch_size, 1), device=code_mask.device, dtype=code_mask.dtype
        )
        code_mask_extended = torch.cat([code_mask, eos_mask], dim=1)  # (B, Lc+1)

        # Add Positional Encoding (Sequence Logic)
        code_h = self.pos_encoder(code_h)

        # Pass through Transformer Encoder
        # PyTorch Transformer expects `src_key_padding_mask` to be True for PAD
        code_padding_mask = ~code_mask_extended.bool()
        code_ctx = self.code_encoder(
            code_h, src_key_padding_mask=code_padding_mask
        )  # (B, Lc+1, D)

        # 3. Prepare Markdown Set (Queries)
        # No Positional Encoding (Set Logic)
        md_padding_mask = ~md_mask.bool()
        md_ctx = self.md_encoder(
            md_h, src_key_padding_mask=md_padding_mask
        )  # (B, Lm, D)

        # 4. Prediction Head (Cross-Attention)
        # Query: Markdown Context (B, Lm, D)
        # Key: Code Context (B, Lc+1, D)
        # Logits = (Q @ K^T) / sqrt(D)

        # (B, Lm, D) @ (B, D, Lc+1) -> (B, Lm, Lc+1)
        logits = torch.matmul(md_ctx, code_ctx.transpose(1, 2)) / self.scale

        # Mask out padded positions in the code sequence
        # We expand the code padding mask to match logits shape
        # code_padding_mask is (B, Lc+1) -> Expand to (B, Lm, Lc+1)
        mask_expanded = code_padding_mask.unsqueeze(1).expand(-1, logits.size(1), -1)

        # Set logits to -inf where code is padding so Softmax ignores them
        logits = logits.masked_fill(mask_expanded, -1e9)

        return logits
