import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from library.config import Config


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        """
        Standard Sinusoidal Positional Encoding.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape (Batch, Seq_Len, Dim)
        Returns:
            Tensor, shape (Batch, Seq_Len, Dim)
        """
        # pe is (Max_Len, 1, Dim), transpose to (1, Max_Len, Dim) for broadcasting
        x = x + self.pe[: x.size(1)].transpose(0, 1)
        return self.dropout(x)


class DCAN(nn.Module):
    def __init__(self):
        """
        Corrected Dual-Context Anchor Network (DC-AN).
        """
        super().__init__()
        self.config = Config()

        input_dim = self.config.EMBEDDING_DIM  # 768
        hidden_dim = self.config.LATENT_DIM  # 512
        nhead = self.config.NHEAD
        num_layers = self.config.NUM_LAYERS
        dropout = self.config.DROPOUT

        # 1. Symmetric Projection Heads
        # Projects MPNet embeddings to a shared latent geometry
        self.code_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.md_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 2. Learnable EOS Token
        # Represents the position "after the last code cell"
        self.eos_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        # 3. Context Encoders
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-Norm usually stabilizes training
        )

        self.code_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.md_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 4. Positional Encoding (Only for Code Sequence)
        # Max len + 2 to account for potential EOS and off-by-one safety
        self.pos_encoder = PositionalEncoding(
            hidden_dim, max_len=self.config.MAX_SEQ_LEN + 10, dropout=dropout
        )

        # Scaling factor for dot product attention
        self.scale = math.sqrt(hidden_dim)

    def forward(self, code_embeddings, markdown_embeddings, code_lens, md_lens):
        """
        Args:
            code_embeddings: (B, Max_Code, 768)
            markdown_embeddings: (B, Max_Md, 768)
            code_lens: (B,) Valid lengths of code sequences
            md_lens: (B,) Valid lengths of markdown sequences

        Returns:
            logits: (B, Max_Md, Max_Code + 1) - Scores for placing MD before each code cell or at EOS.
        """
        batch_size = code_embeddings.size(0)
        max_code_len = code_embeddings.size(1)
        max_md_len = markdown_embeddings.size(1)
        device = code_embeddings.device

        # --- A. Project Embeddings ---
        code_h = self.code_proj(code_embeddings)  # (B, L_c, 512)
        md_h = self.md_proj(markdown_embeddings)  # (B, L_m, 512)

        # --- B. Prepare Code Context (Anchors) ---
        # We construct a sequence of length (Max_Code + 1) to accommodate the EOS token.
        # We dynamically insert the EOS token at the index `code_lens[i]` for each batch item.
        seq_len_with_eos = max_code_len + 1
        code_with_eos = torch.zeros(
            batch_size, seq_len_with_eos, self.config.LATENT_DIM, device=device
        )

        # Create padding mask for Transformer: True indicates padding (ignore)
        code_key_padding_mask = torch.ones(
            batch_size, seq_len_with_eos, device=device, dtype=torch.bool
        )

        # Fill the tensor and mask
        # Note: A loop is used here for clarity and safety with variable lengths.
        # Given batch_size ~64, overhead is negligible compared to Transformer ops.
        for i in range(batch_size):
            c_len = code_lens[i].item()
            if c_len > 0:
                code_with_eos[i, :c_len] = code_h[i, :c_len]

            # Insert EOS token at the end of the valid sequence
            code_with_eos[i, c_len] = self.eos_token

            # Unmask valid positions: 0 to c_len (inclusive, so c_len+1 items)
            code_key_padding_mask[i, : c_len + 1] = False

        # Apply Positional Encoding to Code Sequence
        code_with_eos = self.pos_encoder(code_with_eos)

        # Encode Code Sequence
        code_ctx = self.code_encoder(
            code_with_eos, src_key_padding_mask=code_key_padding_mask
        )

        # --- C. Prepare Markdown Context (Queries) ---
        # Create padding mask for Markdown
        md_key_padding_mask = torch.ones(
            batch_size, max_md_len, device=device, dtype=torch.bool
        )

        for i in range(batch_size):
            m_len = md_lens[i].item()
            if m_len > 0:
                md_key_padding_mask[i, :m_len] = False

        # Encode Markdown Set
        # Note: No Positional Encoding is applied here. This makes it a Set Transformer,
        # allowing the model to treat markdown cells as a bag of queries that can attend
        # to each other to resolve global context/ordering ambiguities.
        md_ctx = self.md_encoder(md_h, src_key_padding_mask=md_key_padding_mask)

        # --- D. Interaction Head ---
        # Compute similarity between Markdown Queries and Code Keys
        # Query: md_ctx (B, M, D)
        # Key: code_ctx (B, C+1, D)
        # Result: (B, M, C+1)
        logits = torch.bmm(md_ctx, code_ctx.transpose(1, 2)) / self.scale

        # --- E. Masking Output ---
        # We must mask the logits corresponding to padded code positions so they don't
        # affect the loss or predictions.
        # Expand code mask to (B, M, C+1)
        mask_expanded = code_key_padding_mask.unsqueeze(1).expand(-1, max_md_len, -1)

        # Fill masked positions with -inf
        logits = logits.masked_fill(mask_expanded, float("-inf"))

        return logits
