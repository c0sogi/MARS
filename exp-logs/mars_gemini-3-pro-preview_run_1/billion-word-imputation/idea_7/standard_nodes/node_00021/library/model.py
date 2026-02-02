import torch
import torch.nn as nn
import math
from library.config import Config


class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding or learnable embeddings.
    Here we use learnable embeddings for simplicity and effectiveness with fixed max length.
    """

    def __init__(self, d_model, max_len=Config.MAX_SEQ_LEN, dropout=Config.DROPOUT):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        batch_size, seq_len, _ = x.size()

        # Create position indices: [0, 1, ..., seq_len-1]
        positions = torch.arange(seq_len, dtype=torch.long, device=x.device).unsqueeze(
            0
        )

        # Add positional embeddings to input embeddings
        x = x + self.pos_embedding(positions)
        return self.dropout(x)


class SyntaxAwareTransformer(nn.Module):
    """
    Transformer-based model for Cloze task with auxiliary syntax supervision.

    Architecture:
    1. Embedding Layer
    2. Shared Transformer Encoder
    3. Three Heads:
       - Localization Head: Binary classification (Is this the gap?)
       - Syntax Head: Multi-class classification (POS Tag)
       - Identification Head: Multi-class classification (Word ID)
    """

    def __init__(self):
        super().__init__()

        self.d_model = Config.EMBED_DIM
        self.vocab_size = Config.VOCAB_SIZE
        self.num_pos_tags = Config.NUM_POS_TAGS

        # 1. Embeddings
        self.embedding = nn.Embedding(
            self.vocab_size, self.d_model, padding_idx=Config.PAD_IDX
        )
        self.pos_encoder = PositionalEncoding(
            self.d_model, max_len=Config.MAX_SEQ_LEN, dropout=Config.DROPOUT
        )

        # 2. Shared Backbone
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=Config.NUM_HEADS,
            dim_feedforward=Config.HIDDEN_DIM,
            dropout=Config.DROPOUT,
            batch_first=True,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=Config.NUM_LAYERS
        )

        # 3. Heads
        # Localization Head: Projects to 1 scalar (logit for sigmoid)
        self.loc_head = nn.Linear(self.d_model, 1)

        # Syntax Head: Projects to number of POS tags
        self.syntax_head = nn.Linear(self.d_model, self.num_pos_tags)

        # Identification Head: Projects to vocabulary size
        # We use a bias-only layer + tied weights if desired, but here we use a full linear layer
        self.id_head = nn.Linear(self.d_model, self.vocab_size)

        self._init_weights()

    def _init_weights(self):
        """Initialize parameters with Xavier/Kaiming initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input_ids, attention_mask=None):
        """
        Args:
            input_ids (Tensor): Shape (batch_size, seq_len)
            attention_mask (Tensor): Shape (batch_size, seq_len). 1 for valid, 0 for pad.

        Returns:
            dict: {
                "loc_logits": (batch_size, seq_len),
                "syntax_logits": (batch_size, seq_len, num_pos_tags),
                "word_logits": (batch_size, seq_len, vocab_size)
            }
        """
        # Create padding mask for Transformer (True where padded)
        # attention_mask is 1 for keep, 0 for pad. src_key_padding_mask needs True for pad.
        if attention_mask is not None:
            src_key_padding_mask = attention_mask == 0
        else:
            src_key_padding_mask = None

        # 1. Embed and Add Position Info
        # Scale embeddings by sqrt(d_model) as per Attention is All You Need
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)

        # 2. Transformer Encoder
        # Output shape: (batch_size, seq_len, d_model)
        hidden_states = self.transformer_encoder(
            x, src_key_padding_mask=src_key_padding_mask
        )

        # 3. Heads

        # Localization: (B, S, 1) -> (B, S)
        loc_logits = self.loc_head(hidden_states).squeeze(-1)

        # Syntax: (B, S, num_tags)
        syntax_logits = self.syntax_head(hidden_states)

        # Identification: (B, S, vocab_size)
        word_logits = self.id_head(hidden_states)

        return {
            "loc_logits": loc_logits,
            "syntax_logits": syntax_logits,
            "word_logits": word_logits,
        }
