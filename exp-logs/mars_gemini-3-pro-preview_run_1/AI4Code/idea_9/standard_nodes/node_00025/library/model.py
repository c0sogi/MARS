import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from library.config import Config


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create constant 'pe' matrix with values dependent on pos and i
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        # x: (Batch, Seq_Len, Dim)
        # Add positional encoding to the input embeddings
        # Slice pe to the current sequence length
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class DualContextAnchorNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = Config

        # Hyperparameters
        self.input_dim = self.config.EMBEDDING_DIM  # 768
        self.latent_dim = self.config.LATENT_DIM  # 512
        self.nhead = self.config.NHEAD  # 8
        self.num_layers = self.config.NUM_LAYERS  # 2
        self.dropout = self.config.DROPOUT  # 0.1

        # 1. Symmetric Projection Heads
        # Maps MPNet embeddings to the latent space
        self.code_proj = nn.Sequential(
            nn.Linear(self.input_dim, self.latent_dim),
            nn.LayerNorm(self.latent_dim),
            nn.GELU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )

        self.md_proj = nn.Sequential(
            nn.Linear(self.input_dim, self.latent_dim),
            nn.LayerNorm(self.latent_dim),
            nn.GELU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )

        # 2. Learnable EOS Token
        # Represents the "End of Notebook" position
        self.eos_token = nn.Parameter(torch.randn(1, 1, self.latent_dim))

        # 3. Context Encoders
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=self.nhead,
            dim_feedforward=self.latent_dim * 4,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
        )

        # Code Encoder: Processes sequence with order (Positional Encoding added in forward)
        self.code_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=self.num_layers
        )

        # Markdown Encoder: Processes set without order (No Positional Encoding)
        # This acts as a Set Transformer
        self.md_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=self.num_layers
        )

        # Positional Encoding for Code
        self.pos_encoder = PositionalEncoding(
            self.latent_dim, max_len=2048, dropout=self.dropout
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize parameters."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # Normal init for EOS token
        nn.init.normal_(self.eos_token, mean=0, std=0.02)

    def forward(
        self,
        code_embeddings,
        code_lens,
        code_padding_mask,
        markdown_embeddings,
        md_lens,
        md_padding_mask,
    ):
        """
        Args:
            code_embeddings: (B, Max_Code, 768)
            code_lens: (B,) - Actual length of code sequences
            code_padding_mask: (B, Max_Code) - True where padding
            markdown_embeddings: (B, Max_MD, 768)
            md_lens: (B,)
            md_padding_mask: (B, Max_MD) - True where padding

        Returns:
            logits: (B, Max_MD, Max_Code + 1)
        """
        batch_size = code_embeddings.size(0)
        device = code_embeddings.device

        # --- 1. Projection ---
        # Project to latent space
        code_feat = self.code_proj(code_embeddings)  # (B, Max_Code, 512)
        md_feat = self.md_proj(markdown_embeddings)  # (B, Max_MD, 512)

        # --- 2. Code Branch (Sequential Context) ---
        # We need to append the EOS token to the end of each VALID code sequence.
        # Since sequences are padded, we can't just concat at the end.
        # We create a new tensor of size L+1.

        max_code_len = code_feat.size(1)
        extended_len = max_code_len + 1

        # Initialize container for extended code sequences
        extended_code = torch.zeros(
            batch_size,
            extended_len,
            self.latent_dim,
            device=device,
            dtype=code_feat.dtype,
        )
        # Initialize extended mask (True = Padding)
        extended_mask = torch.ones(
            batch_size, extended_len, device=device, dtype=torch.bool
        )

        # Copy original code features
        # Note: Padding positions in code_feat are transformed garbage, but they will remain masked.
        extended_code[:, :max_code_len, :] = code_feat
        extended_mask[:, :max_code_len] = code_padding_mask

        # Insert EOS Token at the valid end of each sequence
        # code_lens contains the index where EOS should go (0-indexed length = next index)
        batch_indices = torch.arange(batch_size, device=device)

        # Assign EOS
        # self.eos_token is (1, 1, D) -> squeeze to (D) for assignment
        extended_code[batch_indices, code_lens] = self.eos_token.squeeze(0).squeeze(0)

        # Unmask the EOS position (set to False = Valid)
        extended_mask[batch_indices, code_lens] = False

        # Apply Positional Encoding to Code Sequence
        code_ctx = self.pos_encoder(extended_code)

        # Encode Code Sequence
        # src_key_padding_mask: True for padding positions
        code_out = self.code_encoder(
            code_ctx, src_key_padding_mask=extended_mask
        )  # (B, L+1, 512)

        # --- 3. Markdown Branch (Set Context) ---
        # Encode Markdown Set (No Positional Encoding)
        # This allows the model to learn global relationships (e.g., "I am the only H1")
        md_out = self.md_encoder(
            md_feat, src_key_padding_mask=md_padding_mask
        )  # (B, M, 512)

        # --- 4. Interaction Head ---
        # Compute Logits: Similarity between MD Queries and Code Keys
        # MD: (B, M, D)
        # Code: (B, C+1, D)
        # Logits: (B, M, C+1)
        logits = torch.bmm(md_out, code_out.transpose(1, 2))

        # Mask out logits corresponding to padding in Code
        # extended_mask is (B, C+1). Expand to (B, M, C+1)
        mask_expanded = extended_mask.unsqueeze(1).expand(-1, md_out.size(1), -1)

        # Set padding logits to -inf so Softmax ignores them
        logits = logits.masked_fill(mask_expanded, float("-inf"))

        return logits
