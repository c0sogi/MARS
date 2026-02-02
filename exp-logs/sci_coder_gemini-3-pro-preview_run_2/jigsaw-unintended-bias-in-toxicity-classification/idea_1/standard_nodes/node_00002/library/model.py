import torch
import torch.nn as nn
from library.config import (
    VOCAB_SIZE,
    EMBED_DIM,
    HIDDEN_DIM,
    LSTM_LAYERS,
    BIDIRECTIONAL,
    DROPOUT,
    SPATIAL_DROPOUT,
    IDENTITY_COLUMNS,
)


class SpatialDropout(nn.Module):
    def __init__(self, p):
        super(SpatialDropout, self).__init__()
        self.p = p

    def forward(self, x):
        # x: (batch, seq_len, embed_dim)
        x = x.permute(0, 2, 1)  # (batch, embed_dim, seq_len)
        x = nn.functional.dropout2d(x, p=self.p, training=self.training)
        x = x.permute(0, 2, 1)  # (batch, seq_len, embed_dim)
        return x


class Attention(nn.Module):
    def __init__(self, feature_dim):
        super(Attention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim), nn.Tanh(), nn.Linear(feature_dim, 1)
        )

    def forward(self, x):
        # x: [batch, seq_len, feature_dim]
        weights = self.attention(x)  # [batch, seq_len, 1]
        weights = torch.softmax(weights, dim=1)
        # Weighted sum
        return torch.sum(x * weights, dim=1)


class MultiTaskLSTM(nn.Module):
    """
    A Multi-Task Bidirectional LSTM model for toxicity detection and identity attribute prediction.
    Enhanced with Spatial Dropout and Attention (Cite solution_lesson_node_00001).

    Architecture:
    - Embedding Layer
    - Spatial Dropout
    - Bidirectional LSTM
    - Attention Mechanism
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

        # 2. Spatial Dropout
        self.spatial_dropout = SpatialDropout(SPATIAL_DROPOUT)

        # 3. LSTM Encoder
        self.lstm = nn.LSTM(
            input_size=EMBED_DIM,
            hidden_size=HIDDEN_DIM,
            num_layers=LSTM_LAYERS,
            bidirectional=BIDIRECTIONAL,
            batch_first=True,
        )

        # Calculate the input dimension for the linear heads
        self.lstm_output_dim = HIDDEN_DIM * 2 if BIDIRECTIONAL else HIDDEN_DIM

        # 4. Attention
        self.attention = Attention(self.lstm_output_dim)

        # 5. Dropout
        self.dropout = nn.Dropout(p=DROPOUT)

        # 6. Multi-Task Heads
        self.toxicity_head = nn.Sequential(
            nn.Linear(self.lstm_output_dim, 1), nn.Sigmoid()
        )

        num_identities = len(IDENTITY_COLUMNS)
        self.identity_head = nn.Sequential(
            nn.Linear(self.lstm_output_dim, num_identities), nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: [batch_size, seq_len]

        # Embedding
        embedded = self.embedding(x)
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
