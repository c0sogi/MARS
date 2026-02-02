import math
import torch
import torch.nn as nn
from library.config import Config


class LSTMTagger(nn.Module):
    """
    Bi-LSTM based model for token classification.
    Cite solution_lesson_node_00012: Prefer Bi-LSTMs over Raw Transformers.
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
            bidirectional=Config.BIDIRECTIONAL,
            dropout=Config.DROPOUT if Config.N_LAYERS > 1 else 0,
        )

        lstm_output_dim = (
            Config.HIDDEN_DIM * 2 if Config.BIDIRECTIONAL else Config.HIDDEN_DIM
        )
        self.classifier = nn.Linear(lstm_output_dim, num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: Tensor of shape [batch_size, seq_len] containing token indices.

        Returns:
            logits: Tensor of shape [batch_size, seq_len, num_classes]
        """
        # Embed tokens
        x = self.embedding(input_ids)

        # Pass through LSTM
        x, _ = self.lstm(x)

        # Project to class logits
        logits = self.classifier(x)

        return logits
