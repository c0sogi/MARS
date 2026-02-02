import torch
import torch.nn as nn
import math
from library.config import Config


class DecoupledTransformer(nn.Module):
    """
    Decoupled Localization-Classification Transformer.

    This model processes a sentence with a missing word and predicts:
    1. The location of the missing word (Localization).
    2. The identity of the missing word (Identification).

    It uses a shared Transformer Encoder backbone with two specific output heads.
    """

    def __init__(self, vocab_size):
        super(DecoupledTransformer, self).__init__()

        self.d_model = Config.D_MODEL
        self.vocab_size = vocab_size
        self.max_len = Config.MAX_LEN

        # ----------------------------------------------------------------------
        # 1. Embeddings
        # ----------------------------------------------------------------------
        # Word Embedding: Learnable vector for each token
        # padding_idx=0 assumes PAD_TOKEN is at index 0 (standard in this pipeline)
        self.embedding = nn.Embedding(vocab_size, self.d_model, padding_idx=0)

        # Positional Embedding: Learnable vector for each position
        self.pos_embedding = nn.Embedding(self.max_len, self.d_model)

        self.dropout = nn.Dropout(Config.DROPOUT)

        # ----------------------------------------------------------------------
        # 2. Backbone: Transformer Encoder
        # ----------------------------------------------------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=Config.N_HEADS,
            dim_feedforward=Config.DIM_FEEDFORWARD,
            dropout=Config.DROPOUT,
            activation=Config.ACTIVATION,
            batch_first=True,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=Config.N_LAYERS
        )

        # ----------------------------------------------------------------------
        # 3. Decoupled Heads
        # ----------------------------------------------------------------------
        # Localization Head: Predicts logit P(Gap | Context) for each position
        # Output shape: (batch, seq_len, 1) -> squeeze to (batch, seq_len)
        self.loc_head = nn.Linear(self.d_model, 1)

        # Identification Head: Predicts logit P(Word | Context) for each position
        # Output shape: (batch, seq_len, vocab_size)
        self.id_head = nn.Linear(self.d_model, vocab_size)

        self._init_weights()

    def _init_weights(self):
        """
        Initialize parameters with a uniform distribution.
        """
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.pos_embedding.weight.data.uniform_(-initrange, initrange)
        self.loc_head.bias.data.zero_()
        self.loc_head.weight.data.uniform_(-initrange, initrange)
        self.id_head.bias.data.zero_()
        self.id_head.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, attention_mask=None):
        """
        Forward pass of the model.

        Args:
            src (torch.Tensor): Input token indices. Shape (batch_size, seq_len).
            attention_mask (torch.Tensor, optional): Mask where 1 indicates valid token
                                                     and 0 indicates padding.
                                                     Shape (batch_size, seq_len).

        Returns:
            loc_logits (torch.Tensor): Logits for gap existence. Shape (batch_size, seq_len).
            id_logits (torch.Tensor): Logits for word prediction. Shape (batch_size, seq_len, vocab_size).
        """
        batch_size, seq_len = src.size()

        # Safety Truncation: Ensure sequence length does not exceed architectural limit
        if seq_len > self.max_len:
            src = src[:, : self.max_len]
            if attention_mask is not None:
                attention_mask = attention_mask[:, : self.max_len]
            seq_len = self.max_len

        # 1. Embeddings
        # Create position indices [0, 1, ..., seq_len-1]
        positions = torch.arange(0, seq_len, dtype=torch.long, device=src.device)
        positions = positions.unsqueeze(0).expand(batch_size, seq_len)

        # Combine word and pos embeddings
        src_emb = self.embedding(src) * math.sqrt(self.d_model)
        pos_emb = self.pos_embedding(positions)
        x = self.dropout(src_emb + pos_emb)

        # 2. Backbone
        # Prepare padding mask for PyTorch Transformer
        # PyTorch expects: True for PADDED (ignore), False for REAL
        # Input attention_mask: 1 for REAL, 0 for PADDED
        if attention_mask is not None:
            src_key_padding_mask = attention_mask == 0
        else:
            src_key_padding_mask = None

        encoded = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)

        # 3. Heads
        # Localization: Output (batch, seq_len)
        loc_logits = self.loc_head(encoded).squeeze(-1)

        # Identification: Output (batch, seq_len, vocab_size)
        id_logits = self.id_head(encoded)

        return loc_logits, id_logits
