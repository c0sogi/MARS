import math
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: [Batch, Seq, Dim]
        # output: [Batch, Seq, Dim]
        # Slicing pe to [1, Seq, Dim]
        return x + self.pe[: x.size(1), :].unsqueeze(0)


class DC_AN(nn.Module):
    """
    Corrected Dual-Context Anchor Network (DC-AN).
    Aligns contextualized markdown queries (Set) against contextualized code anchors (Sequence).
    """

    def __init__(self, config=None):
        super(DC_AN, self).__init__()
        self.config = config if config else Config

        # 1. Symmetric Projection Towers
        # Maps MPNet (768) -> Latent (512)
        self.code_proj = nn.Linear(self.config.INPUT_DIM, self.config.LATENT_DIM)
        self.md_proj = nn.Linear(self.config.INPUT_DIM, self.config.LATENT_DIM)

        # 2. Positional Encoding (Only for Code Sequence)
        self.pos_encoder = SinusoidalPositionalEncoding(
            self.config.LATENT_DIM, max_len=2048
        )

        # 3. Transformer Encoders
        # Shared architecture hyperparameters
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.LATENT_DIM,
            nhead=self.config.N_HEADS,
            dim_feedforward=self.config.LATENT_DIM * 4,
            dropout=self.config.DROPOUT,
            batch_first=True,
            norm_first=True,  # Usually stabilizes training
        )

        # Code Branch: Contextualized Skeleton
        self.code_transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=self.config.NUM_LAYERS
        )

        # Markdown Branch: Contextualized Queries (Set Transformer)
        # We use a separate instance with same config
        self.md_transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=self.config.NUM_LAYERS
        )

        # 4. Learnable EOS Token
        # Represents the "End of Notebook" position
        self.eos_token = nn.Parameter(torch.randn(1, 1, self.config.LATENT_DIM))

        # Dropout
        self.dropout = nn.Dropout(self.config.DROPOUT)

        # Initialization
        self._init_weights()

    def _init_weights(self):
        # Initialize linear layers and EOS
        nn.init.xavier_uniform_(self.code_proj.weight)
        nn.init.xavier_uniform_(self.md_proj.weight)
        nn.init.normal_(self.eos_token, mean=0.0, std=0.02)

    def forward(self, code_emb, code_lens, md_emb, md_mask):
        """
        Args:
            code_emb: [Batch, Max_Code_Len, Input_Dim]
            code_lens: [Batch] - Valid lengths of code sequences
            md_emb: [Batch, Max_MD_Len, Input_Dim]
            md_mask: [Batch, Max_MD_Len] - True for valid tokens, False for padding

        Returns:
            logits: [Batch, Max_MD_Len, Max_Code_Len + 1]
        """
        B = code_emb.size(0)
        device = code_emb.device

        # Ensure code_lens is on the correct device for masking logic
        if code_lens.device != device:
            code_lens = code_lens.to(device)

        # ==========================================
        # 1. Projection & Embedding
        # ==========================================
        c_feat = self.code_proj(code_emb)  # [B, L, 512]
        m_feat = self.md_proj(md_emb)  # [B, M, 512]

        # ==========================================
        # 2. Code Branch (Sequence + EOS)
        # ==========================================
        L = c_feat.size(1)

        # Create container for Code + EOS: [B, L+1, 512]
        c_feat_eos = torch.zeros(
            B, L + 1, self.config.LATENT_DIM, device=device, dtype=c_feat.dtype
        )

        # Copy original code features
        c_feat_eos[:, :L, :] = c_feat

        # Insert EOS token dynamically at the end of each valid sequence
        # We use advanced indexing: [0..B-1, code_lens]
        batch_indices = torch.arange(B, device=device)
        c_feat_eos[batch_indices, code_lens, :] = self.eos_token

        # Create Padding Mask for Code
        # We want to mask out indices > code_lens
        # Indices are 0 to L (total L+1 positions)
        # Valid indices: 0 to code_lens (inclusive)
        seq_indices = torch.arange(L + 1, device=device).unsqueeze(0)  # [1, L+1]
        # valid_mask: True where index <= code_len
        valid_code_mask = seq_indices <= code_lens.unsqueeze(1)
        # src_key_padding_mask expects True for PADDING (ignored)
        code_padding_mask = ~valid_code_mask

        # Apply Positional Encoding to Code Sequence
        c_feat_eos = self.pos_encoder(c_feat_eos)
        c_feat_eos = self.dropout(c_feat_eos)

        # Contextualize Code
        c_context = self.code_transformer(
            c_feat_eos, src_key_padding_mask=code_padding_mask
        )

        # ==========================================
        # 3. Markdown Branch (Set)
        # ==========================================
        # No positional encoding for Set Transformer
        m_feat = self.dropout(m_feat)

        # Invert md_mask for Transformer (Input: True=Valid -> Transformer: True=Pad)
        md_padding_mask = ~md_mask

        # Contextualize Markdown
        m_context = self.md_transformer(m_feat, src_key_padding_mask=md_padding_mask)

        # ==========================================
        # 4. Interaction Head (Cross-Attention Logits)
        # ==========================================
        # Query: Markdown [B, M, 512]
        # Key: Code [B, L+1, 512]
        # Logits: [B, M, L+1]

        logits = torch.bmm(m_context, c_context.transpose(1, 2))

        # Mask out logits corresponding to padded code positions
        # code_padding_mask is [B, L+1] (True=Pad)
        # Expand to [B, M, L+1]
        if m_context.size(1) > 0:
            logit_mask = code_padding_mask.unsqueeze(1).expand(
                -1, m_context.size(1), -1
            )
            logits = logits.masked_fill(logit_mask, -1e9)

        return logits
