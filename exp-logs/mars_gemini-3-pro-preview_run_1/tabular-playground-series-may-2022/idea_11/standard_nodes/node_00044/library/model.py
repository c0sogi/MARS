import torch
import torch.nn as nn
import torch.nn.functional as F


class GEGLU(nn.Module):
    """
    Gated Linear Unit variant where the input is split in half,
    and one half is activated by GELU and multiplied by the other.
    Reference: Shazeer et al., "GLU Variants Improve Transformer" (2020)
    """

    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * F.gelu(gate)


class ResDeGUTLayer(nn.Module):
    """
    Custom Transformer Encoder Layer incorporating GEGLU in the feed-forward block.
    Uses Pre-Layer Normalization (NormFirst).
    """

    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        # Feed-forward block with GEGLU
        # Project to 2 * dim_feedforward because GEGLU splits the dimension
        self.linear1 = nn.Linear(d_model, dim_feedforward * 2)
        self.geglu = GEGLU()
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Pre-LN Self Attention
        src_norm = self.norm1(src)
        attn_output, _ = self.self_attn(
            src_norm,
            src_norm,
            src_norm,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )
        src = src + self.dropout1(attn_output)

        # Pre-LN Feed Forward (GEGLU)
        src_norm = self.norm2(src)
        ff_out = self.linear1(src_norm)
        ff_out = self.geglu(ff_out)
        ff_out = self.dropout(ff_out)
        ff_out = self.linear2(ff_out)
        src = src + self.dropout2(ff_out)

        return src


class ResDeGUT(nn.Module):
    """
    Denoising Granular Unified Transformer (DeGUT).
    Refactored to remove the Residual MLP branch and include full reconstruction.
    Features:
    - Shared Embeddings (Linear for Num, Entity for Seq)
    - Unified Transformer with GEGLU
    - Masked Reconstruction Heads for BOTH Sequence and Numerical features
    """

    def __init__(self, num_features, seq_len, vocab_size, config):
        super().__init__()
        self.config = config
        self.d_model = config.HIDDEN_DIM

        # --- Shared Embeddings ---
        # Linear Tokenization for numerical features: (B, N) -> (B, N, D)
        self.num_tokenizer = nn.Linear(1, self.d_model)
        # Entity Embeddings for sequence features: (B, L) -> (B, L, D)
        self.seq_embedding = nn.Embedding(vocab_size, self.d_model)
        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.d_model))

        # --- Positional Encoding ---
        # Total sequence length = 1 (CLS) + num_features + seq_len
        total_len = 1 + num_features + seq_len
        self.pos_embedding = nn.Parameter(torch.randn(1, total_len, self.d_model))

        # --- Deep Transformer ---
        # Stack of custom layers with GEGLU
        self.layers = nn.ModuleList(
            [
                ResDeGUTLayer(
                    d_model=self.d_model,
                    nhead=config.NHEAD,
                    dim_feedforward=self.d_model * 4,
                    dropout=config.DROPOUT,
                )
                for _ in range(config.NUM_LAYERS)
            ]
        )

        # --- Heads ---
        # 1. Classification Head (on CLS token)
        self.classifier = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(self.d_model, 1),
        )

        # 2. Sequence Reconstruction Head (on Sequence tokens)
        self.seq_recon = nn.Linear(self.d_model, vocab_size)

        # 3. Numerical Reconstruction Head (on Numerical tokens)
        # Projects (B, N_num, D) -> (B, N_num, 1) -> Squeeze to (B, N_num)
        self.num_recon = nn.Linear(self.d_model, 1)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x_num, x_seq):
        batch_size = x_num.size(0)

        # 1. Embedding
        # Numerical: (B, N_num) -> (B, N_num, 1) -> (B, N_num, D)
        x_num_emb = self.num_tokenizer(x_num.unsqueeze(-1))
        # Sequence: (B, Seq_len) -> (B, Seq_len, D)
        x_seq_emb = self.seq_embedding(x_seq)

        # 2. Transformer Input Construction
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        # Concatenate: [CLS] + Num Tokens + Seq Tokens
        x_deep = torch.cat((cls_tokens, x_num_emb, x_seq_emb), dim=1)

        # Add Positional Encoding
        seq_len_total = x_deep.size(1)
        x_deep = x_deep + self.pos_embedding[:, :seq_len_total, :]

        # Pass through Transformer Layers
        for layer in self.layers:
            x_deep = layer(x_deep)

        # 3. Outputs
        # CLS Token -> Classification
        cls_out = x_deep[:, 0, :]
        logits = self.classifier(cls_out)

        # Numerical Tokens -> Reconstruction
        # Indices: 1 to 1 + N_num
        num_len = x_num.size(1)
        num_out = x_deep[:, 1 : 1 + num_len, :]
        num_recon_preds = self.num_recon(num_out).squeeze(-1)

        # Sequence Tokens -> Reconstruction
        # Indices: 1 + N_num to End
        seq_out = x_deep[:, 1 + num_len :, :]
        seq_recon_logits = self.seq_recon(seq_out)

        return logits, seq_recon_logits, num_recon_preds
