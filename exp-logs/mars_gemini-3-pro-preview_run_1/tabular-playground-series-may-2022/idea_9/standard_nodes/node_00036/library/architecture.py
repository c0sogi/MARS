import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from library.config import Config


class GEGLU(nn.Module):
    """
    Gated Linear Unit with GELU activation.
    References:
    - GLU Variants Improve Transformer (Shazeer, 2020)
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()
        # Project to 2 * dim_out to split into gate and value
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class DiGUTLayer(nn.Module):
    """
    A single Transformer Encoder layer using GEGLU in the FeedForward block.
    Uses Pre-LayerNorm configuration.
    """

    def __init__(self, hidden_dim, num_heads, forward_dim, dropout, attention_dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(hidden_dim)
        # FeedForward with GEGLU
        # Structure: Input -> Norm -> GEGLU_Block -> Dropout -> Residual
        self.ff = nn.Sequential(
            GEGLU(hidden_dim, forward_dim),
            nn.Dropout(dropout),
            nn.Linear(forward_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Attention Block
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.dropout1(attn_out)

        # FeedForward Block
        x_norm = self.norm2(x)
        ff_out = self.ff(x_norm)
        x = x + ff_out

        return x


class DeGUT(nn.Module):
    """
    Denoising Granular Unified Transformer (DeGUT).

    Architecture:
    1. Granular Input:
       - Numerical features -> Linear Projection + Feature Identity Embedding
       - Sequence features -> Entity Embedding + Positional Embedding
       - [CLS] Token prepended
       - [MASK] Token for denoising task
    2. Transformer Encoder: Stack of layers with GEGLU.
    3. Dual Heads:
       - Target Head: MLP on [CLS] for state prediction.
       - Reconstruction Heads: Predict original values of masked tokens.
    """

    def __init__(
        self,
        num_numerical_features: int,
        vocab_size: int,
        sequence_length: int,
        config: Config,
    ):
        super().__init__()
        self.config = config
        dim = config.HIDDEN_DIM

        # ---------------------------------------------------------------------
        # 1. Granular Embeddings
        # ---------------------------------------------------------------------

        # Numerical: Project scalar to vector + Add Feature Identity
        self.num_proj = nn.Linear(1, dim)
        self.num_id_emb = nn.Parameter(torch.zeros(1, num_numerical_features, dim))
        nn.init.normal_(self.num_id_emb, std=0.02)

        # Sequence: Entity Embedding + Positional Embedding
        self.seq_emb = nn.Embedding(vocab_size, dim)
        self.seq_pos_emb = nn.Parameter(torch.zeros(1, sequence_length, dim))
        nn.init.normal_(self.seq_pos_emb, std=0.02)

        # [CLS] Token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.cls_token, std=0.02)

        # [MASK] Token (Shared for both modalities)
        self.mask_emb = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.mask_emb, std=0.02)

        # Dropout applied after embeddings
        self.emb_dropout = nn.Dropout(config.DROPOUT)

        # ---------------------------------------------------------------------
        # 2. Transformer Encoder
        # ---------------------------------------------------------------------
        self.layers = nn.ModuleList(
            [
                DiGUTLayer(
                    hidden_dim=dim,
                    num_heads=config.NUM_HEADS,
                    forward_dim=config.FORWARD_DIM,
                    dropout=config.DROPOUT,
                    attention_dropout=config.ATTENTION_DROPOUT,
                )
                for _ in range(config.NUM_LAYERS)
            ]
        )

        self.final_norm = nn.LayerNorm(dim)

        # ---------------------------------------------------------------------
        # 3. Heads
        # ---------------------------------------------------------------------

        # Target Head
        self.target_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(dim, 1),
        )

        # Reconstruction Heads (Cite solution_lesson_node_00032)
        # Numerical: Regress the scalar value
        self.num_recon_head = nn.Linear(dim, 1)
        # Sequence: Predict the character class
        self.seq_recon_head = nn.Linear(dim, vocab_size)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x_num, x_seq, mask_num=None, mask_seq=None):
        """
        Args:
            x_num: (Batch, Num_Feats)
            x_seq: (Batch, Seq_Len)
            mask_num: (Batch, Num_Feats) Boolean mask (True = Masked)
            mask_seq: (Batch, Seq_Len) Boolean mask (True = Masked)
        """
        B = x_num.shape[0]

        # --- Embedding Numerical ---
        num_emb = self.num_proj(x_num.unsqueeze(-1))

        # Apply Masking to Numerical Embeddings
        if mask_num is not None:
            mask_emb_expanded = self.mask_emb.expand(B, num_emb.size(1), -1)
            num_emb = torch.where(mask_num.unsqueeze(-1), mask_emb_expanded, num_emb)

        num_emb = num_emb + self.num_id_emb

        # --- Embedding Sequence ---
        seq_emb = self.seq_emb(x_seq)

        # Apply Masking to Sequence Embeddings
        if mask_seq is not None:
            mask_emb_expanded = self.mask_emb.expand(B, seq_emb.size(1), -1)
            seq_emb = torch.where(mask_seq.unsqueeze(-1), mask_emb_expanded, seq_emb)

        seq_emb = seq_emb + self.seq_pos_emb

        # --- Unified Sequence Construction ---
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, num_emb, seq_emb), dim=1)

        x = self.emb_dropout(x)

        # --- Transformer Encoder ---
        for layer in self.layers:
            x = layer(x)

        x = self.final_norm(x)

        # --- Output Heads ---

        # 1. Target Prediction
        cls_out = x[:, 0, :]
        target_logits = self.target_head(cls_out)

        # 2. Reconstruction Prediction
        # Extract tokens corresponding to features
        num_tokens = x[:, 1 : 1 + num_emb.size(1), :]
        seq_tokens = x[:, 1 + num_emb.size(1) :, :]

        num_preds = self.num_recon_head(num_tokens).squeeze(-1)
        seq_preds = self.seq_recon_head(seq_tokens)

        return target_logits, num_preds, seq_preds
