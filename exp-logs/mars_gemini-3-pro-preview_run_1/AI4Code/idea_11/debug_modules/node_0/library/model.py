import torch
import torch.nn as nn
import math
from library.config import Config


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (Batch, Seq_Len, Dim)
        seq_len = x.size(1)
        # Slice pe to the current sequence length and add to x
        return x + self.pe[:seq_len, :].unsqueeze(0)


class DualContextAnchorNetwork(nn.Module):
    def __init__(self):
        super(DualContextAnchorNetwork, self).__init__()

        # Hyperparameters
        self.input_dim = 768  # Output dim of all-mpnet-base-v2
        self.hidden_dim = Config.HIDDEN_DIM
        self.nhead = Config.NHEAD
        self.num_layers = Config.NUM_LAYERS
        self.dropout = Config.DROPOUT

        # 1. Projections
        # Map Code and Markdown embeddings to a shared latent space
        self.code_projection = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )

        self.md_projection = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )

        # 2. Positional Encoding (Applied only to Code sequence)
        self.pos_encoder = PositionalEncoding(self.hidden_dim)

        # 3. Context Encoders
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.nhead,
            dim_feedforward=self.hidden_dim * 4,
            dropout=self.dropout,
            batch_first=True,
        )

        # Code Encoder: Processes the ordered sequence of code cells
        self.code_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=self.num_layers
        )

        # Markdown Encoder: Processes the set of markdown cells (Set Transformer)
        # Note: We share the same layer architecture but separate weights.
        # No positional encoding is added to inputs, making it permutation invariant.
        self.md_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=self.num_layers
        )

        # 4. Learnable EOS Token
        # Represents the position "after the last code cell"
        self.eos_token = nn.Parameter(torch.randn(1, 1, self.hidden_dim))

        # 5. Scaling factor for dot-product attention
        self.scale = math.sqrt(self.hidden_dim)

    def forward(
        self, code_embeddings, code_mask, code_lens, md_embeddings, md_mask, md_lens
    ):
        """
        Args:
            code_embeddings: (Batch, Code_Len, 768)
            code_mask: (Batch, Code_Len) - Boolean mask, True indicates padding.
            code_lens: (Batch,) - Actual lengths of code sequences.
            md_embeddings: (Batch, MD_Len, 768)
            md_mask: (Batch, MD_Len) - Boolean mask, True indicates padding.
            md_lens: (Batch,) - Actual lengths of markdown sets.

        Returns:
            logits: (Batch, MD_Len, Code_Len + 1)
        """
        batch_size = code_embeddings.size(0)
        device = code_embeddings.device

        # --- 1. Projection ---
        code_h = self.code_projection(code_embeddings)  # (B, L_code, H)
        md_h = self.md_projection(md_embeddings)  # (B, L_md, H)

        # --- 2. Contextualization ---

        # Code Branch: Add Positional Info -> Transformer
        code_h = self.pos_encoder(code_h)
        code_ctx = self.code_encoder(code_h, src_key_padding_mask=code_mask)

        # Markdown Branch: No Positional Info -> Transformer (Set interactions)
        md_ctx = self.md_encoder(md_h, src_key_padding_mask=md_mask)

        # --- 3. Dynamic EOS Insertion ---
        # We construct a new tensor of shape (B, L_code + 1, H)
        # The EOS token is inserted at index `code_lens[i]` for each batch element.

        max_code_len = code_ctx.size(1)
        extended_len = max_code_len + 1

        # Initialize with zeros (padding)
        code_ctx_eos = torch.zeros(
            batch_size, extended_len, self.hidden_dim, device=device
        )

        # Copy valid code context
        code_ctx_eos[:, :max_code_len, :] = code_ctx

        # Prepare EOS token for scattering
        # Target indices must be broadcastable to src
        # code_lens shape (B) -> (B, 1, H)
        target_indices = (
            code_lens.view(batch_size, 1, 1).expand(-1, 1, self.hidden_dim).to(device)
        )
        eos_expanded = self.eos_token.expand(batch_size, -1, -1)

        # Scatter EOS into the correct positions
        code_ctx_eos = code_ctx_eos.scatter(1, target_indices, eos_expanded)

        # --- 4. Update Code Mask for EOS ---
        # New mask shape: (B, L_code + 1)
        code_mask_eos = torch.ones(
            batch_size, extended_len, dtype=torch.bool, device=device
        )
        # Copy original mask
        code_mask_eos[:, :max_code_len] = code_mask

        # Mark the EOS position as False (Valid/Not Padding)
        target_indices_mask = code_lens.view(batch_size, 1).to(device)
        false_tensor = torch.zeros(batch_size, 1, dtype=torch.bool, device=device)
        code_mask_eos = code_mask_eos.scatter(1, target_indices_mask, false_tensor)

        # --- 5. Interaction Head (Dot Product) ---
        # Query: Markdown Context (B, L_md, H)
        # Key: Code Context + EOS (B, L_code + 1, H)

        # Compute raw scores: (B, L_md, L_code + 1)
        logits = torch.bmm(md_ctx, code_ctx_eos.transpose(1, 2)) / self.scale

        # --- 6. Masking ---
        # We must mask positions in logits corresponding to padding in code_ctx_eos
        # code_mask_eos is (B, L_code + 1). Expand to match logits.
        # Shape: (B, 1, L_code + 1) -> (B, L_md, L_code + 1)
        logit_mask = code_mask_eos.unsqueeze(1).expand(-1, logits.size(1), -1)

        # Apply mask (set padding positions to -inf)
        logits = logits.masked_fill(logit_mask, float("-inf"))

        return logits
