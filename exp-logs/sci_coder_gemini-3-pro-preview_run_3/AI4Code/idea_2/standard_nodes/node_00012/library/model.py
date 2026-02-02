import torch
import torch.nn as nn
from library.config import Config


class DSAPR(nn.Module):
    """
    Deep Sequence-Aware Position Regressor (DSAPR).

    This model predicts the relative rank (0 to 1) of a Markdown cell (Query)
    within a sequence of Code cells (Context).

    Architecture:
    1. Input: Markdown Embedding (Query) + Sequence of Code Embeddings (Context).
    2. Positional Encoding: Added to Code embeddings to preserve order.
    3. Transformer Encoder: Attends to context to learn semantic relationships.
    4. Regression Head: MLP predicting the scalar rank.
    """

    def __init__(self):
        super(DSAPR, self).__init__()

        # Hyperparameters from Config
        self.embed_dim = Config.EMBED_DIM
        self.max_len = Config.MAX_SEQ_LEN
        self.n_layers = Config.TRANSFORMER_LAYERS
        self.n_heads = Config.TRANSFORMER_HEADS
        self.dropout_rate = Config.DROPOUT
        self.forward_expansion = Config.FORWARD_EXPANSION

        # 1. Positional Embeddings for the Context (Code Sequence)
        # We use a learnable parameter. Shape: (1, max_len, embed_dim)
        # Initialized with small random values.
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.max_len, self.embed_dim) * 0.02
        )

        # 2. Transformer Encoder
        # batch_first=True ensures input shape is (Batch, Seq_Len, Dim)
        # norm_first=True (Pre-LN) is generally more stable for convergence
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.n_heads,
            dim_feedforward=self.embed_dim * self.forward_expansion,
            dropout=self.dropout_rate,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=self.n_layers
        )

        # 3. Regression Head
        # Projects the Query output state to a single scalar [0, 1]
        self.regressor = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.embed_dim // 2, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for linear layers using Xavier Uniform."""
        for p in self.regressor.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, query, context, mask):
        """
        Forward pass of the DSAPR model.

        Args:
            query (Tensor): Markdown cell embeddings. Shape (B, embed_dim).
            context (Tensor): Code cell embeddings sequence. Shape (B, max_len, embed_dim).
            mask (Tensor): Mask for context (1 for valid, 0 for pad). Shape (B, max_len).

        Returns:
            Tensor: Predicted relative rank. Shape (B,).
        """
        batch_size = query.size(0)

        # --- 1. Prepare Inputs ---

        # Expand query to sequence format: (B, 1, embed_dim)
        query_seq = query.unsqueeze(1)

        # Add positional encodings to context
        # We slice pos_embedding to match the current context length (usually max_len)
        # Broadcasting handles the batch dimension.
        seq_len = context.size(1)
        context_pos = context + self.pos_embedding[:, :seq_len, :]

        # Concatenate Query and Context along sequence dimension
        # Result: [Query, Code_1, Code_2, ..., Code_N]
        # Shape: (B, 1 + seq_len, embed_dim)
        src = torch.cat([query_seq, context_pos], dim=1)

        # --- 2. Prepare Attention Mask ---

        # The transformer needs to know which positions are padding.
        # We also need to ensure the Query token is never masked.
        # Create a mask for the query token (all 1s/valid)
        query_mask = torch.ones((batch_size, 1), device=mask.device, dtype=mask.dtype)

        # Concatenate with context mask: (B, 1 + seq_len)
        full_mask = torch.cat([query_mask, mask], dim=1)

        # Convert to BoolTensor for PyTorch Transformer
        # In PyTorch Transformer, True indicates the value should be IGNORED (i.e., padding).
        # Our input mask has 1=Valid, 0=Pad.
        # So we want True where mask == 0.
        src_key_padding_mask = full_mask == 0

        # --- 3. Transformer Pass ---

        # Output shape: (B, 1 + seq_len, embed_dim)
        encoded_output = self.transformer_encoder(
            src, src_key_padding_mask=src_key_padding_mask
        )

        # --- 4. Regression ---

        # We only care about the output state of the Query token (index 0)
        # This token has attended to all code cells and gathered context.
        query_output = encoded_output[:, 0, :]  # Shape: (B, embed_dim)

        # Predict rank
        prediction = self.regressor(query_output)  # Shape: (B, 1)

        return prediction.squeeze(-1)  # Shape: (B,)
