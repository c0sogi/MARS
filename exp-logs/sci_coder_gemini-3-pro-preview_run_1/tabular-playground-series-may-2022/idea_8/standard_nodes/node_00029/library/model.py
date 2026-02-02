import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class GEGLU(nn.Module):
    """
    Gated Linear Unit with GELU activation.
    Projects input to 2*dim, splits, and applies gating.
    Reference: GLU Variants Improve Transformer (Shazeer et al., 2020)
    """

    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * F.gelu(gate)


class DeGUTEncoderLayer(nn.Module):
    """
    Custom Transformer Encoder Layer using GEGLU and Pre-Norm.
    """

    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        # Implementation of GEGLU FFN
        # Project to 2 * dim_feedforward to allow splitting
        self.linear1 = nn.Linear(d_model, dim_feedforward * 2)
        self.act = GEGLU()
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        # Pre-Norm Architecture
        # 1. Self-Attention Block
        src2 = self.norm1(src)
        src2, _ = self.self_attn(
            src2,
            src2,
            src2,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            is_causal=is_causal,
        )
        src = src + self.dropout1(src2)

        # 2. Feed-Forward Block (GEGLU)
        src2 = self.norm2(src)
        src2 = self.linear1(src2)
        src2 = self.act(src2)
        src2 = self.dropout(src2)
        src2 = self.linear2(src2)
        src = src + self.dropout2(src2)

        return src


class GranularEmbedding(nn.Module):
    """
    Projects numerical and sequence features into a unified latent space.
    Implements 'Linear Feature Tokenization' for numerical data and handles
    token replacement for the denoising objective.
    """

    def __init__(self, num_feats, vocab_size, d_model, max_len=Config.MAX_SEQ_LEN):
        super().__init__()

        # 1. Numerical Feature Projection
        # We use distinct linear projections for each numerical feature.
        # Implemented via broadcasting: weight (1, N_num, D), bias (1, N_num, D)
        # This preserves the specific semantic meaning of each physical measurement.
        self.num_weights = nn.Parameter(torch.randn(1, num_feats, d_model) * 0.02)
        self.num_biases = nn.Parameter(torch.zeros(1, num_feats, d_model))

        # Learnable [MASK] token for numerical features
        # Used to replace values when mask_num is True
        self.num_mask_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # 2. Sequence Feature Embedding
        # Standard embedding layer. Sequence masking is handled by token ID replacement
        # in the collator, so this layer just looks up the ID (including the mask ID).
        self.seq_embedding = nn.Embedding(vocab_size, d_model)

        # 3. Special Tokens
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # 4. Positional Embedding
        # Learnable position embeddings for the concatenated sequence ([CLS] + Num + Seq)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(Config.DROPOUT)

    def forward(self, x_num, x_seq, mask_num=None):
        """
        Args:
            x_num: (B, N_num) float tensor
            x_seq: (B, N_seq) long tensor (already has mask token IDs where appropriate)
            mask_num: (B, N_num) boolean tensor (True = mask this value)
        """
        B = x_num.shape[0]

        # --- Process Numerical ---
        # Project scalar features to vectors: val * weight + bias
        # (B, N_num, 1) * (1, N_num, D) + (1, N_num, D) -> (B, N_num, D)
        out_num = x_num.unsqueeze(-1) * self.num_weights + self.num_biases

        # Apply Masking to Numerical Features
        if mask_num is not None:
            # Expand mask to (B, N_num, D)
            mask_expanded = mask_num.unsqueeze(-1).expand_as(out_num)
            # Replace masked embeddings with the learnable num_mask_token
            out_num = torch.where(mask_expanded, self.num_mask_token, out_num)

        # --- Process Sequence ---
        # (B, N_seq) -> (B, N_seq, D)
        out_seq = self.seq_embedding(x_seq)

        # --- Concatenate ---
        # Prepend [CLS] token
        # (1, 1, D) -> (B, 1, D)
        cls_tokens = self.cls_token.expand(B, -1, -1)

        # Combined: [CLS, Num_1..Num_N, Seq_1..Seq_M]
        x = torch.cat([cls_tokens, out_num, out_seq], dim=1)

        # --- Add Positional Info ---
        seq_len = x.size(1)
        # Add positional embeddings (broadcast over batch)
        # We slice to the current sequence length
        if seq_len <= self.pos_embedding.size(1):
            x = x + self.pos_embedding[:, :seq_len, :]
        else:
            # Fallback if input exceeds max_len (though Config should prevent this)
            x = x + self.pos_embedding[:, : self.pos_embedding.size(1), :]

        x = self.layer_norm(x)
        x = self.dropout(x)

        return x


class DeGUTModel(nn.Module):
    """
    Denoising Granular Unified Transformer (DeGUT).

    A Transformer-based model that learns the joint distribution of manufacturing
    control codes and machine states via a multi-task objective:
    1. Binary Classification (Target)
    2. Denoising Autoencoding (Reconstruction of masked inputs)
    """

    def __init__(self, num_feats, vocab_size):
        super().__init__()

        self.d_model = Config.D_MODEL

        # Granular Embedding Layer
        self.embedding = GranularEmbedding(
            num_feats=num_feats,
            vocab_size=vocab_size,
            d_model=self.d_model,
            max_len=Config.MAX_SEQ_LEN,
        )

        # Transformer Encoder
        # Using custom layer with GEGLU activation
        encoder_layer = DeGUTEncoderLayer(
            d_model=self.d_model,
            nhead=Config.N_HEADS,
            dim_feedforward=Config.DIM_FEEDFORWARD,
            dropout=Config.DROPOUT,
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=Config.N_LAYERS)

        # --- Output Heads ---

        # 1. Classification Head (from CLS token)
        self.head_cls = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(self.d_model, 1),
        )

        # 2. Reconstruction Head (Numerical)
        # Projects vectors back to scalar values
        self.head_num = nn.Linear(self.d_model, 1)

        # 3. Reconstruction Head (Sequence)
        # Projects vectors back to vocabulary logits
        self.head_seq = nn.Linear(self.d_model, vocab_size)

        self._init_weights()

    def _init_weights(self):
        """Initialize parameters with Xavier Uniform"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self, num_features, seq_features, mask_num=None, mask_seq=None, **kwargs
    ):
        """
        Forward pass of the DeGUT model.

        Args:
            num_features: (B, N_num)
            seq_features: (B, N_seq)
            mask_num: (B, N_num) boolean mask for numerical features
            mask_seq: (B, N_seq) boolean mask for sequence features (unused here,
                      as the collator handles sequence ID replacement)
            **kwargs: Catch-all for other args

        Returns:
            dict: containing 'logits_cls', 'pred_num', 'pred_seq'
        """

        # Embed inputs (handles numerical masking internally)
        x = self.embedding(num_features, seq_features, mask_num)

        # Pass through Transformer Encoder
        x = self.encoder(x)

        # Split outputs based on input structure
        # x structure: [CLS, Num_1...Num_N, Seq_1...Seq_M]
        cls_out = x[:, 0, :]

        num_len = num_features.shape[1]
        num_out = x[:, 1 : 1 + num_len, :]
        seq_out = x[:, 1 + num_len :, :]

        # Generate Predictions
        logits_cls = self.head_cls(cls_out)  # (B, 1)

        pred_num = self.head_num(num_out).squeeze(-1)  # (B, N_num)
        pred_seq = self.head_seq(seq_out)  # (B, N_seq, Vocab)

        return {"logits_cls": logits_cls, "pred_num": pred_num, "pred_seq": pred_seq}
