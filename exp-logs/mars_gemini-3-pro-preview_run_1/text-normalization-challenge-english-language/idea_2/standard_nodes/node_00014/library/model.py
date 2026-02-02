import math
import torch
import torch.nn as nn
from library.config import Config


class LSTMTagger(nn.Module):
    """
    Bi-LSTM based model for token classification.
    """

    def __init__(self, vocab_size: int, num_classes: int, pad_token_id: int = 0):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=Config.EMBED_DIM,
            padding_idx=pad_token_id,
        )

        self.lstm = nn.LSTM(
            input_size=Config.EMBED_DIM,
            hidden_size=Config.HIDDEN_DIM,
            num_layers=Config.N_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=Config.DROPOUT if Config.N_LAYERS > 1 else 0,
        )

        self.dropout = nn.Dropout(Config.DROPOUT)

        # Bidirectional LSTM outputs hidden_size * 2
        self.classifier = nn.Linear(Config.HIDDEN_DIM * 2, num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: Tensor of shape [batch_size, seq_len]
        Returns:
            logits: Tensor of shape [batch_size, seq_len, num_classes]
        """
        # Embed
        x = self.embedding(input_ids)

        # LSTM
        # output shape: [batch, seq, hidden*2]
        x, _ = self.lstm(x)

        x = self.dropout(x)

        # Classify
        logits = self.classifier(x)

        return logits
