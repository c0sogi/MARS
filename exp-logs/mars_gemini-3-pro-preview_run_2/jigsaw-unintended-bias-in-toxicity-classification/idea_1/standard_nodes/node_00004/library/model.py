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


class SpatialDropout(nn.Dropout2d):
    def forward(self, x):
        x = x.permute(0, 2, 1)  # (N, L, C) -> (N, C, L)
        x = super(SpatialDropout, self).forward(x)
        x = x.permute(0, 2, 1)  # (N, C, L) -> (N, L, C)
        return x


class AttentionPooling(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.Tanh(), nn.Linear(in_dim, 1)
        )

    def forward(self, x):
        # x: [batch, seq_len, hidden_dim]
        w = self.attention(x)  # [batch, seq_len, 1]
        w = torch.softmax(w, dim=1)
        return torch.sum(x * w, dim=1)


class MultiTaskLSTM(nn.Module):
    """
    A Multi-Task Bidirectional LSTM model for toxicity detection and identity attribute prediction.

    Architecture:
    - Embedding Layer
    - Spatial Dropout (Cite solution_lesson_node_00002)
    - Bidirectional LSTM
    - Attention Pooling (Cite solution_lesson_node_00002)
    - Shared Dropout
    - Toxicity Head (Binary Classification)
    - Identity Head (Multi-label Classification)
    """

    def __init__(self):
        super(MultiTaskLSTM, self).__init__()

        # 1. Embedding Layer
        self.embedding = nn.Embedding(
            num_embeddings=VOCAB_SIZE, embedding_dim=EMBED_DIM, padding_idx=0
        )

        # Spatial Dropout on embeddings
        self.spatial_dropout = SpatialDropout(p=DROPOUT)

        # 2. LSTM Encoder
        self.lstm = nn.LSTM(
            input_size=EMBED_DIM,
            hidden_size=HIDDEN_DIM,
            num_layers=LSTM_LAYERS,
            bidirectional=BIDIRECTIONAL,
            batch_first=True,
        )

        # Calculate the input dimension for the linear heads
        self.lstm_output_dim = HIDDEN_DIM * 2 if BIDIRECTIONAL else HIDDEN_DIM

        # Attention Pooling
        self.attention = AttentionPooling(self.lstm_output_dim)

        # 3. Linear Dropout
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
        embedded = self.embedding(x)

        # Spatial Dropout
        embedded = self.spatial_dropout(embedded)

        # LSTM
        lstm_out, _ = self.lstm(embedded)

        # Attention Pooling
        pooled = self.attention(lstm_out)

        # Dropout
        pooled = self.dropout(pooled)

        # Heads
        toxicity_pred = self.toxicity_head(pooled)
        identity_pred = self.identity_head(pooled)

        return toxicity_pred, identity_pred
