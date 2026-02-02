import torch
import torch.nn as nn
from library.config import Config


class NumericalProjector(nn.Module):
    """
    Projects continuous numerical features into the embedding space
    to create a 'Numerical Context Token'.
    Structure: Linear -> GELU -> Linear
    """

    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, x):
        # x shape: (Batch_Size, Num_Features)
        return self.net(x)


class SequenceEmbedder(nn.Module):
    """
    Embeds sequence tokens and adds learnable positional encodings.
    """

    def __init__(self, vocab_size, embed_dim, max_len):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_len, embed_dim)

        # Register position IDs as a buffer so they are part of the state_dict
        # and move to device automatically, but are not updated by optimizer.
        self.register_buffer("pos_ids", torch.arange(max_len))

    def forward(self, x):
        # x shape: (Batch_Size, Seq_Len)
        b, l = x.shape

        # Get token embeddings
        tok_emb = self.token_embedding(x)  # (B, L, D)

        # Add positional encodings (broadcasting across batch)
        # We slice pos_ids to the current sequence length l
        pos_emb = self.pos_embedding(self.pos_ids[:l])  # (L, D)

        return tok_emb + pos_emb


class UnifiedTransformer(nn.Module):
    """
    Unified Multi-Modal Transformer.
    Treats numerical data as a context token prepended to the character sequence.
    """

    def __init__(self):
        super().__init__()

        # ----------------------------------------------------------------------
        # Configuration
        # ----------------------------------------------------------------------
        num_features = len(Config.NUMERICAL_FEATURES)
        vocab_size = Config.VOCAB_SIZE
        embed_dim = Config.EMBED_DIM
        max_len = Config.MAX_SEQ_LEN
        nhead = Config.NUM_HEADS
        num_layers = Config.NUM_TRANSFORMER_LAYERS
        dropout = Config.TRANSFORMER_DROPOUT
        mlp_layers = Config.MLP_HIDDEN_LAYERS
        mlp_dropout = Config.MLP_DROPOUT

        # ----------------------------------------------------------------------
        # 1. Feature Encoders
        # ----------------------------------------------------------------------
        self.num_projector = NumericalProjector(num_features, embed_dim)
        self.seq_embedder = SequenceEmbedder(vocab_size, embed_dim, max_len)

        # ----------------------------------------------------------------------
        # 2. Transformer Backbone
        # ----------------------------------------------------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN often stabilizes training
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # ----------------------------------------------------------------------
        # 3. Classification Head
        # ----------------------------------------------------------------------
        # Input: Concat of [Transformed Numerical Token] + [Pooled Sequence]
        head_input_dim = embed_dim * 2

        layers = []
        in_dim = head_input_dim

        for hidden_dim in mlp_layers:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(mlp_dropout))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, 1))
        self.head = nn.Sequential(*layers)

    def forward(self, numerical, sequence):
        """
        Args:
            numerical (torch.Tensor): (Batch, Num_Features)
            sequence (torch.Tensor): (Batch, Seq_Len) - Integer tokens
        Returns:
            torch.Tensor: Probabilities (Batch,)
        """
        # 1. Embed Inputs
        # Numerical -> (B, D) -> (B, 1, D)
        num_emb = self.num_projector(numerical).unsqueeze(1)

        # Sequence -> (B, L, D)
        seq_emb = self.seq_embedder(sequence)

        # 2. Early Fusion
        # Concatenate along the sequence dimension: [Num_Token, Seq_Token_1, ..., Seq_Token_L]
        x = torch.cat([num_emb, seq_emb], dim=1)  # (B, 1+L, D)

        # 3. Create Padding Mask
        # The Transformer expects `src_key_padding_mask` where True indicates padding (to be ignored).
        # Numerical token (index 0) is never padding.
        # Sequence tokens (indices 1+) are padding if value is 0.

        B, L = sequence.shape
        seq_padding_mask = sequence == 0  # (B, L)
        num_padding_mask = torch.zeros(
            (B, 1), dtype=torch.bool, device=x.device
        )  # (B, 1)

        # Combined mask: (B, 1+L)
        src_key_padding_mask = torch.cat([num_padding_mask, seq_padding_mask], dim=1)

        # 4. Transformer Pass
        # Output: (B, 1+L, D)
        x_out = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        # 5. Feature Extraction & Pooling
        # Extract the transformed numerical context (Index 0)
        num_context = x_out[:, 0, :]  # (B, D)

        # Extract sequence outputs (Indices 1 to End)
        seq_context = x_out[:, 1:, :]  # (B, L, D)

        # Masked Global Average Pooling for Sequence
        # We zero out padding positions to ensure they don't contribute to the mean
        mask_expanded = seq_padding_mask.unsqueeze(-1)  # (B, L, 1)
        seq_context_masked = seq_context.masked_fill(mask_expanded, 0.0)

        # Calculate valid lengths for division
        valid_lens = (~seq_padding_mask).sum(dim=1, keepdim=True).float()  # (B, 1)
        valid_lens = torch.clamp(valid_lens, min=1.0)  # Prevent div by zero

        seq_pooled = seq_context_masked.sum(dim=1) / valid_lens  # (B, D)

        # 6. Final Fusion & Classification
        fused = torch.cat([num_context, seq_pooled], dim=1)  # (B, 2*D)
        logits = self.head(fused)  # (B, 1)

        # Return probabilities
        return torch.sigmoid(logits).squeeze(1)
