import torch
import torch.nn as nn
from library.config import (
    VOCAB_SIZE,
    EMBED_DIM,
    HIDDEN_DIM,
    LSTM_LAYERS,
    BIDIRECTIONAL,
    DROPOUT,
    IDENTITY_COLUMNS,
)


class MultiTaskLSTM(nn.Module):
    """
    A Multi-Task Bidirectional LSTM model for toxicity detection and identity attribute prediction.

    Architecture:
    - Embedding Layer
    - Bidirectional LSTM
    - Global Max Pooling
    - Shared Dropout
    - Toxicity Head (Binary Classification)
    - Identity Head (Multi-label Classification)
    """

    def __init__(self):
        super(MultiTaskLSTM, self).__init__()

        # 1. Embedding Layer
        # Padding index is 0 as per the Tokenizer in data_loader.py
        self.embedding = nn.Embedding(
            num_embeddings=VOCAB_SIZE, embedding_dim=EMBED_DIM, padding_idx=0
        )

        # 2. LSTM Encoder
        self.lstm = nn.LSTM(
            input_size=EMBED_DIM,
            hidden_size=HIDDEN_DIM,
            num_layers=LSTM_LAYERS,
            bidirectional=BIDIRECTIONAL,
            batch_first=True,
        )

        # Calculate the input dimension for the linear heads
        # If bidirectional, the output dimension is hidden_dim * 2
        self.lstm_output_dim = HIDDEN_DIM * 2 if BIDIRECTIONAL else HIDDEN_DIM

        # 3. Dropout
        self.dropout = nn.Dropout(p=DROPOUT)

        # 4. Multi-Task Heads

        # Toxicity Head: Predicts the main toxicity target (scalar)
        self.toxicity_head = nn.Sequential(
            nn.Linear(self.lstm_output_dim, 1), nn.Sigmoid()
        )

        # Identity Head: Predicts the presence of 9 identity attributes
        num_identities = len(IDENTITY_COLUMNS)
        self.identity_head = nn.Sequential(
            nn.Linear(self.lstm_output_dim, num_identities), nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len) containing token indices.

        Returns:
            toxicity_pred (torch.Tensor): Shape (batch_size, 1) - Toxicity probabilities.
            identity_pred (torch.Tensor): Shape (batch_size, num_identities) - Identity probabilities.
        """
        # x shape: [batch_size, seq_len]

        # Embedding
        # shape: [batch_size, seq_len, embed_dim]
        embedded = self.embedding(x)

        # LSTM
        # output shape: [batch_size, seq_len, lstm_output_dim]
        # hidden/cell states are ignored
        lstm_out, _ = self.lstm(embedded)

        # Global Max Pooling
        # Take the maximum value over the sequence dimension (dim=1)
        # shape: [batch_size, lstm_output_dim]
        pooled, _ = torch.max(lstm_out, dim=1)

        # Dropout
        pooled = self.dropout(pooled)

        # Heads
        toxicity_pred = self.toxicity_head(pooled)
        identity_pred = self.identity_head(pooled)

        return toxicity_pred, identity_pred
