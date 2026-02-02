import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from library.config import PROJECTION_DIM, TRANSFORMER_LAYERS, N_HEADS, DROPOUT


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

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
        # pe: (Max_Len, Dim) -> slice to (1, Seq_Len, Dim)
        # Ensure we don't exceed max_len
        seq_len = x.size(1)
        if seq_len > self.pe.size(0):
            # Fallback or error, but max_len=5000 is sufficient for this task
            pass
        x = x + self.pe[:seq_len, :].unsqueeze(0)
        return self.dropout(x)


class SymmetricProjector(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class CAAN(nn.Module):
    def __init__(self):
        super().__init__()

        # Fixed input dim from SentenceTransformer backbone
        self.input_dim = 384
        self.hidden_dim = PROJECTION_DIM

        # Projectors for heterogeneous modalities
        self.code_projector = SymmetricProjector(
            self.input_dim, self.hidden_dim, DROPOUT
        )
        self.md_projector = SymmetricProjector(self.input_dim, self.hidden_dim, DROPOUT)

        # Learnable EOS Token to represent "end of notebook" position
        self.eos_token = nn.Parameter(torch.randn(1, 1, self.hidden_dim))

        # Contextual Anchor Encoder
        self.pos_encoder = PositionalEncoding(self.hidden_dim, DROPOUT, max_len=2048)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=N_HEADS,
            dim_feedforward=self.hidden_dim * 4,
            dropout=DROPOUT,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=TRANSFORMER_LAYERS
        )

        self.dropout = nn.Dropout(DROPOUT)

        self._init_weights()

    def _init_weights(self):
        # Initialize EOS token
        nn.init.normal_(self.eos_token, mean=0, std=0.02)

        # Initialize Linear layers
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, code_emb, code_mask, md_emb, md_mask=None):
        """
        Args:
            code_emb: (Batch, Max_Code_Len, 384)
            code_mask: (Batch, Max_Code_Len) - Boolean, True where valid token
            md_emb: (Batch, Max_Md_Len, 384)
            md_mask: (Batch, Max_Md_Len) - Boolean, True where valid (unused in computation)

        Returns:
            logits: (Batch, Max_Md_Len, Max_Code_Len + 1)
                    Scores for placing markdown before code_i, or at EOS.
        """
        B, L_c, _ = code_emb.shape
        device = code_emb.device

        # 1. Project Embeddings to latent space
        code_feat = self.code_projector(code_emb)  # (B, L_c, 512)
        md_feat = self.md_projector(md_emb)  # (B, L_m, 512)

        # 2. Construct Contextual Anchors with EOS
        # We need to insert the EOS token immediately after the last valid code token for each batch item.
        # code_mask sums to the length of valid tokens (N). We place EOS at index N.

        lengths = code_mask.sum(dim=1).long()  # (B,)

        # Create a container for Code + EOS. Size is L_c + 1 to accommodate EOS even if L_c is full.
        extended_code_feat = torch.zeros(B, L_c + 1, self.hidden_dim, device=device)
        extended_mask = torch.zeros(B, L_c + 1, dtype=torch.bool, device=device)

        # Copy valid code features.
        # Since pad_sequence pads at the end, valid tokens are at 0..N-1.
        extended_code_feat[:, :L_c, :] = code_feat
        extended_mask[:, :L_c] = code_mask

        # Insert EOS token at the correct position (index = length)
        batch_indices = torch.arange(B, device=device)
        eos_expanded = self.eos_token.expand(B, 1, -1).squeeze(1)  # (B, 512)

        extended_code_feat[batch_indices, lengths, :] = eos_expanded
        extended_mask[batch_indices, lengths] = True

        # 3. Apply Positional Encoding
        # Input to transformer: (B, L_c+1, 512)
        x = self.pos_encoder(extended_code_feat)

        # 4. Transformer Encoding
        # src_key_padding_mask requires True for padded (ignored) positions.
        # extended_mask is True for valid positions.
        padding_mask = ~extended_mask

        contextual_code = self.transformer_encoder(x, src_key_padding_mask=padding_mask)

        # 5. Attention Mechanism (Classification Head)
        # Query: Markdown (B, L_m, 512)
        # Key: Contextual Code (B, L_c+1, 512)
        # Logits = Q @ K^T / sqrt(d)

        logits = torch.bmm(md_feat, contextual_code.transpose(1, 2))
        logits = logits / math.sqrt(self.hidden_dim)

        # 6. Mask Padded Keys
        # We must mask out positions in the logits corresponding to padding in code sequence.
        # padding_mask shape: (B, L_c+1) -> Expand to (B, L_m, L_c+1)
        mask_expanded = padding_mask.unsqueeze(1).expand(-1, md_feat.size(1), -1)

        # Fill padded positions with -inf so Softmax ignores them
        logits = logits.masked_fill(mask_expanded, float("-inf"))

        return logits
