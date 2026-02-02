import torch
import torch.nn as nn
from library.config import Config


class EssayModel(nn.Module):
    """
    LSTM-based Regressor for Essay Scoring.
    Replaces the Deep Averaging Network (DAN) to capture sequence information.
    Cite solution_lesson_node_00001: Addressing the need for sequence-aware models.

    Architecture:
    1. Embedding Layer
    2. Bidirectional LSTM
    3. Global Average Pooling (Masked)
    4. MLP Head
    """

    def __init__(self, config=Config):
        super(EssayModel, self).__init__()

        self.vocab_size = config.VOCAB_SIZE
        self.embed_dim = config.EMBED_DIM
        self.hidden_dim = config.HIDDEN_DIM
        self.output_dim = config.OUTPUT_DIM
        self.dropout_prob = config.DROPOUT_PROB
        self.lstm_layers = config.LSTM_LAYERS
        self.bidirectional = config.BIDIRECTIONAL

        self.padding_idx = 0

        # 1. Embedding
        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embed_dim,
            padding_idx=self.padding_idx,
        )

        # 2. LSTM
        self.lstm = nn.LSTM(
            input_size=self.embed_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.lstm_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
            dropout=self.dropout_prob if self.lstm_layers > 1 else 0,
        )

        # Calculate LSTM output dimension
        self.lstm_out_dim = self.hidden_dim * (2 if self.bidirectional else 1)

        # 3. MLP Head
        self.regressor = nn.Sequential(
            nn.Linear(self.lstm_out_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(self.hidden_dim, self.output_dim),
        )

    def forward(self, input_ids):
        # input_ids: (batch_size, seq_len)

        # Embed
        embedded = self.embedding(input_ids)

        # LSTM
        # We rely on masking later, letting LSTM process padding is acceptable
        # given we pool with a mask.
        lstm_out, _ = self.lstm(embedded)
        # lstm_out: (batch_size, seq_len, lstm_out_dim)

        # Masking
        mask = (input_ids != self.padding_idx).float().unsqueeze(-1)

        # Global Average Pooling (Masked)
        sum_embeddings = torch.sum(lstm_out * mask, dim=1)
        token_counts = torch.sum(mask, dim=1).clamp(min=1e-9)
        mean_embeddings = sum_embeddings / token_counts

        # Regress
        output = self.regressor(mean_embeddings)

        return output
