import torch
import torch.nn as nn
import math
from library.config import Config


class PositionalEncoding(nn.Module):
    """
    Standard Sinusoidal Positional Encoding.
    Injects information about the relative or absolute position of the tokens in the sequence.
    """

    def __init__(self, d_model, max_len=2048):
        super().__init__()
        # Create a long enough 'pe' matrix with position indices
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not a learnable parameter, but part of state_dict)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        # Slice the cached PE matrix to the current sequence length
        return x + self.pe[:, : x.size(1), :]


class DCAN(nn.Module):
    """
    Corrected Dual-Context Anchor Network (DC-AN).
    Aligns a set of permutation-invariant Markdown queries against a sequence of
    contextualized Code anchors.
    """

    def __init__(self):
        super().__init__()
        self.hidden_dim = Config.HIDDEN_DIM
        self.backbone_dim = 768  # Output dim of all-mpnet-base-v2

        # ----------------------------------------------------------------------
        # 1. Projection Heads
        # ----------------------------------------------------------------------
        # Symmetric projection to map heterogeneous inputs to shared latent space
        self.code_proj = nn.Sequential(
            nn.Linear(self.backbone_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT),
        )

        self.md_proj = nn.Sequential(
            nn.Linear(self.backbone_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT),
        )

        # ----------------------------------------------------------------------
        # 2. Context Encoders
        # ----------------------------------------------------------------------
        # Code Branch: Sequential Context (needs Positional Encoding)
        self.pos_encoder = PositionalEncoding(self.hidden_dim, max_len=2048)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=Config.NHEAD,
            dim_feedforward=self.hidden_dim * 4,
            dropout=Config.DROPOUT,
            batch_first=True,
            norm_first=True,  # Pre-Norm usually stabilizes training
        )

        self.code_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=Config.NUM_ENCODER_LAYERS
        )

        # Markdown Branch: Set Context (Permutation Invariant -> No Positional Encoding)
        self.md_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=Config.NUM_ENCODER_LAYERS
        )

        # ----------------------------------------------------------------------
        # 3. Special Tokens
        # ----------------------------------------------------------------------
        # Learnable EOS token to represent "End of Notebook" (after the last code cell)
        self.eos_token = nn.Parameter(torch.randn(1, 1, self.hidden_dim))

        self._init_weights()

    def _init_weights(self):
        """Initialize parameters for stability."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        code_embeddings,
        markdown_embeddings,
        code_mask,
        markdown_mask,
        code_lens,
        markdown_lens,
        **kwargs
    ):
        """
        Forward pass of the DC-AN.

        Args:
            code_embeddings: (Batch, MaxCode, 768)
            markdown_embeddings: (Batch, MaxMD, 768)
            code_mask: (Batch, MaxCode) - BoolTensor, True indicates padding.
            markdown_mask: (Batch, MaxMD) - BoolTensor, True indicates padding.
            code_lens: (Batch,) - LongTensor, actual number of code cells.
            markdown_lens: (Batch,) - LongTensor, actual number of markdown cells.

        Returns:
            logits: (Batch, MaxMD, MaxCode + 1)
                    Scores representing the likelihood of a markdown cell appearing
                    before code cell i (for i < N) or after the last code cell (i == N).
        """
        B = code_embeddings.size(0)
        device = code_embeddings.device

        # ----------------------------------------------------------------------
        # 1. Projection
        # ----------------------------------------------------------------------
        c_emb = self.code_proj(code_embeddings)  # (B, Lc, H)
        m_emb = self.md_proj(markdown_embeddings)  # (B, Lm, H)

        # ----------------------------------------------------------------------
        # 2. Dynamic EOS Insertion (Code Branch)
        # ----------------------------------------------------------------------
        # We create a new sequence of length Lc + 1.
        # The EOS token is inserted at the index `code_lens` for each batch item.
        Lc = c_emb.size(1)

        # Initialize container. We copy existing embeddings.
        # Shape: (B, Lc + 1, H)
        c_emb_eos = torch.zeros(B, Lc + 1, self.hidden_dim, device=device)
        c_emb_eos[:, :Lc, :] = c_emb

        # Insert EOS token at the end of the valid sequence
        batch_indices = torch.arange(B, device=device)
        # code_lens indices point to the first padding position (or Lc if full)
        # We place EOS there.
        c_emb_eos[batch_indices, code_lens] = self.eos_token.squeeze(0)

        # Update Mask
        # New mask shape: (B, Lc + 1)
        c_mask_eos = torch.ones(B, Lc + 1, dtype=torch.bool, device=device)
        c_mask_eos[:, :Lc] = code_mask
        # Unmask the position where we placed EOS (it is now valid)
        c_mask_eos[batch_indices, code_lens] = False

        # ----------------------------------------------------------------------
        # 3. Contextualization
        # ----------------------------------------------------------------------
        # Code: Add Position Info -> Transformer
        c_emb_eos = self.pos_encoder(c_emb_eos)
        c_context = self.code_encoder(
            c_emb_eos, src_key_padding_mask=c_mask_eos
        )  # (B, Lc+1, H)

        # Markdown: Set Transformer (No Positional Encoding)
        # We still provide the padding mask to ignore pad tokens in self-attention
        m_context = self.md_encoder(
            m_emb, src_key_padding_mask=markdown_mask
        )  # (B, Lm, H)

        # ----------------------------------------------------------------------
        # 4. Interaction (Cross-Attention)
        # ----------------------------------------------------------------------
        # We compute the dot product between every Markdown query and every Code anchor.
        # m_context: (B, Lm, H)
        # c_context.transpose: (B, H, Lc+1)
        # logits: (B, Lm, Lc+1)
        logits = torch.matmul(m_context, c_context.transpose(1, 2))

        # ----------------------------------------------------------------------
        # 5. Output Masking
        # ----------------------------------------------------------------------
        # We must ensure the model does not attend to/predict padding positions in the code sequence.
        # c_mask_eos is True for padding. We set those logits to -inf.
        # Broadcast mask: (B, 1, Lc+1)
        logits = logits.masked_fill(c_mask_eos.unsqueeze(1), -float("inf"))

        return logits
