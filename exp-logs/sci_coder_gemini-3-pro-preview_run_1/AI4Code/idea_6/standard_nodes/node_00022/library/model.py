import torch
import torch.nn as nn
import math
from library.config import Config


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (Batch, Seq_Len, Dim)
        # Slicing pe to the current sequence length
        return x + self.pe[:, : x.size(1), :]


class ProjectionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ProjectionHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, x):
        return self.net(x)


class DualContextAnchorNetwork(nn.Module):
    def __init__(self):
        super(DualContextAnchorNetwork, self).__init__()

        self.input_dim = Config.INPUT_DIM
        self.hidden_dim = Config.HIDDEN_DIM
        self.nhead = Config.NHEAD
        self.num_layers = Config.NUM_LAYERS

        # 1. Symmetric Projection Towers
        # Maps 768 -> 512
        self.code_proj = ProjectionHead(self.input_dim, self.hidden_dim)
        self.md_proj = ProjectionHead(self.input_dim, self.hidden_dim)

        # 2. Positional Encoding (Code only)
        # We use a large max_len (5000) to cover any realistic notebook cell count
        self.pos_encoder = PositionalEncoding(self.hidden_dim, max_len=5000)

        # 3. End-of-Notebook Token
        # Represents the position after the last code cell
        self.eon_token = nn.Parameter(torch.randn(1, 1, self.hidden_dim))

        # 4. Context Transformers
        # Standard Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.nhead,
            dim_feedforward=self.hidden_dim * 4,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )

        # Code Encoder: Processes sequential structure
        self.code_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=self.num_layers
        )

        # Markdown Encoder: Processes set structure (Permutation Invariant)
        self.md_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=self.num_layers
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize parameters with Xavier Uniform"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, code_embeddings, markdown_embeddings, code_mask, markdown_mask):
        """
        Forward pass of the DC-AN.

        Args:
            code_embeddings (Tensor): (Batch, N_code, 768)
            markdown_embeddings (Tensor): (Batch, N_md, 768)
            code_mask (Tensor): (Batch, N_code) - Boolean, True=Valid, False=Pad
            markdown_mask (Tensor): (Batch, N_md) - Boolean, True=Valid, False=Pad

        Returns:
            logits (Tensor): (Batch, N_md, N_code + 1)
                             Scores representing the likelihood of a markdown cell
                             following each code cell (or the EoN token).
        """
        batch_size = code_embeddings.size(0)

        # --- 1. Projection ---
        # Project heterogeneous inputs to shared latent space
        # Shape: (B, N, 512)
        h_code = self.code_proj(code_embeddings)
        h_md = self.md_proj(markdown_embeddings)

        # --- 2. Prepare Code Sequence (Add EoN Token) ---
        # Expand learnable EoN token to batch size
        eon = self.eon_token.expand(batch_size, -1, -1)

        # Concatenate EoN to the end of the code sequence
        # Shape: (B, N_code + 1, 512)
        h_code = torch.cat([h_code, eon], dim=1)

        # Update Code Mask to include the valid EoN position
        # Shape: (B, N_code + 1)
        eon_mask = torch.ones(
            (batch_size, 1), device=code_mask.device, dtype=torch.bool
        )
        full_code_mask = torch.cat([code_mask, eon_mask], dim=1)

        # --- 3. Positional Encoding (Code Only) ---
        # Inject order information into the code skeleton
        h_code = self.pos_encoder(h_code)

        # --- 4. Contextualization ---
        # Create padding masks for Transformer (True = Ignore/Pad)
        # We invert our boolean masks (where True = Valid)
        code_padding_mask = ~full_code_mask
        md_padding_mask = ~markdown_mask

        # Process Code (Sequential Context)
        # Shape: (B, N_code + 1, 512)
        h_code_ctx = self.code_encoder(h_code, src_key_padding_mask=code_padding_mask)

        # Process Markdown (Set Context)
        # Note: No positional encoding is added to h_md, making it permutation invariant
        # Shape: (B, N_md, 512)
        h_md_ctx = self.md_encoder(h_md, src_key_padding_mask=md_padding_mask)

        # --- 5. Interaction Head (Dot Product Attention) ---
        # Compute affinity between Contextualized Queries (MD) and Contextualized Keys (Code)
        # Query: (B, N_md, 512)
        # Key:   (B, N_code + 1, 512) -> Transpose -> (B, 512, N_code + 1)
        # Logits: (B, N_md, N_code + 1)
        logits = torch.bmm(h_md_ctx, h_code_ctx.transpose(1, 2))

        # Scale dot product
        logits = logits / math.sqrt(self.hidden_dim)

        # --- 6. Masking Invalid Anchor Positions ---
        # Mask out logits corresponding to padded code cells so they don't affect Softmax/Loss
        # Expand mask to match logits shape: (B, N_md, N_code + 1)
        mask_expanded = full_code_mask.unsqueeze(1).expand(-1, logits.size(1), -1)

        # Apply mask: Set invalid positions to -inf
        logits = logits.masked_fill(~mask_expanded, float("-inf"))

        return logits
