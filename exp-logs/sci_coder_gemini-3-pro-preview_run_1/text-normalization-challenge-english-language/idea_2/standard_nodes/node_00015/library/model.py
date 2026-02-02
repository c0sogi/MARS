import math
import torch
import torch.nn as nn
from library.config import Config


class TransformerTagger(nn.Module):
    """
    Transformer-based model for token classification.
    """

    def __init__(self, vocab_size: int, num_classes: int, pad_token_id: int = 0):
        super().__init__()
        self.pad_token_id = pad_token_id

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=Config.EMBED_DIM,
            padding_idx=pad_token_id,
        )

        # Learned Positional Embedding
        self.pos_embedding = nn.Embedding(Config.MAX_LEN, Config.EMBED_DIM)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=Config.EMBED_DIM,
            nhead=Config.N_HEADS,
            dim_feedforward=Config.FF_DIM,
            dropout=Config.DROPOUT,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=Config.N_LAYERS
        )

        self.dropout = nn.Dropout(Config.DROPOUT)
        self.classifier = nn.Linear(Config.EMBED_DIM, num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: Tensor of shape [batch_size, seq_len]
        Returns:
            logits: Tensor of shape [batch_size, seq_len, num_classes]
        """
        batch_size, seq_len = input_ids.size()

        # Create padding mask (True where padding exists)
        src_key_padding_mask = input_ids == self.pad_token_id

        # Embed
        x = self.embedding(input_ids)

        # Add Positional Embeddings
        # Create positions [0, 1, ..., seq_len-1]
        positions = (
            torch.arange(0, seq_len, device=input_ids.device)
            .unsqueeze(0)
            .expand(batch_size, seq_len)
        )
        x = x + self.pos_embedding(positions)

        x = self.dropout(x)

        # Transformer Encoder
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)

        # Classify
        logits = self.classifier(x)

        return logits
