import math
import torch
import torch.nn as nn
from library.config import Config


class PositionalEncoding(nn.Module):
    """
    Injects some information about the relative or absolute position of the tokens
    in the sequence. The positional encodings have the same dimension as the embeddings,
    so that the two can be summed.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create a long enough 'pe' matrix with position indices
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Calculate the division term for sine and cosine functions
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not a learnable parameter, but part of state_dict)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        # Add positional encoding to embeddings
        # Slice pe to the current sequence length
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerTagger(nn.Module):
    """
    Transformer Encoder-based model for token classification.
    Consists of an Embedding layer, Positional Encoding, Transformer Encoder stack,
    and a Linear classification head.
    """

    def __init__(self, vocab_size: int, num_classes: int, pad_token_id: int = 0):
        super().__init__()

        self.d_model = Config.EMBED_DIM
        self.pad_token_id = pad_token_id

        # 1. Embedding Layer
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=self.d_model,
            padding_idx=pad_token_id,
        )

        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(
            d_model=self.d_model, max_len=Config.MAX_LEN, dropout=Config.DROPOUT
        )

        # 3. Transformer Encoder Backbone
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=Config.N_HEADS,
            dim_feedforward=Config.FF_DIM,
            dropout=Config.DROPOUT,
            batch_first=True,  # Important: inputs are [batch, seq, feature]
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=Config.N_LAYERS
        )

        # 4. Classification Head
        self.classifier = nn.Linear(self.d_model, num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """
        Initialize weights for linear layers and embeddings.
        """
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.classifier.bias.data.zero_()
        self.classifier.weight.data.uniform_(-initrange, initrange)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: Tensor of shape [batch_size, seq_len] containing token indices.

        Returns:
            logits: Tensor of shape [batch_size, seq_len, num_classes]
        """
        # Create padding mask
        # src_key_padding_mask should be True for positions to be ignored (padded)
        src_key_padding_mask = input_ids == self.pad_token_id

        # Embed tokens and scale by sqrt(d_model) as per "Attention is All You Need"
        x = self.embedding(input_ids) * math.sqrt(self.d_model)

        # Add positional info
        x = self.pos_encoder(x)

        # Pass through Transformer Encoder
        # Output shape: [batch_size, seq_len, d_model]
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)

        # Project to class logits
        logits = self.classifier(x)

        return logits
