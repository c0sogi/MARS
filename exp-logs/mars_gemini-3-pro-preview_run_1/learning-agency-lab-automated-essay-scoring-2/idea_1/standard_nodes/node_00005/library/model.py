import torch
import torch.nn as nn
from library.config import Config


class LSTMRegressor(nn.Module):
    """
    Bidirectional LSTM Regressor for essay scoring.

    Architecture:
    1. Embedding Layer
    2. Bidirectional LSTM
    3. Global Average + Max Pooling
    4. MLP Head
    """

    def __init__(self, config=Config):
        super(LSTMRegressor, self).__init__()

        self.vocab_size = config.VOCAB_SIZE
        self.embed_dim = config.EMBED_DIM
        self.hidden_dim = config.HIDDEN_DIM
        self.output_dim = config.OUTPUT_DIM
        self.dropout_prob = config.DROPOUT_PROB
        self.padding_idx = 0

        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embed_dim,
            padding_idx=self.padding_idx,
        )

        self.lstm = nn.LSTM(
            input_size=self.embed_dim,
            hidden_size=self.hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=self.dropout_prob,
        )

        # Input to regressor is (Hidden * 2 directions * 2 poolings)
        self.regressor = nn.Sequential(
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(self.hidden_dim, self.output_dim),
        )

    def forward(self, input_ids):
        # Calculate lengths for packing (must be on CPU)
        lengths = torch.sum(input_ids != self.padding_idx, dim=1).cpu()
        lengths = torch.clamp(lengths, min=1)

        embedded = self.embedding(input_ids)

        # Pack sequence
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False
        )

        packed_output, _ = self.lstm(packed)

        # Unpack
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        # output: (Batch, Seq, Hidden*2)

        # Masking
        mask = (input_ids != self.padding_idx).float().unsqueeze(-1)  # (B, L, 1)

        # Avg Pooling
        sum_embeddings = torch.sum(output * mask, dim=1)
        seq_lengths = torch.sum(mask, dim=1)
        seq_lengths = torch.clamp(seq_lengths, min=1e-9)
        avg_pool = sum_embeddings / seq_lengths

        # Max Pooling
        # Mask padding positions with large negative value
        output_masked = output.clone()
        output_masked[mask.expand_as(output) == 0] = -1e9
        max_pool, _ = torch.max(output_masked, dim=1)

        # Concatenate
        combined = torch.cat([avg_pool, max_pool], dim=1)

        return self.regressor(combined)
