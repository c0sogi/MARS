import torch
import torch.nn as nn
from library.config import Config


class SiameseLSTM(nn.Module):
    """
    Siamese Bi-Directional LSTM for Chatbot Arena preference prediction.
    Cite solution_lesson_node_00001: Replaces GAP with sequence-aware LSTM.
    Cite solution_lesson_node_00002: Uses safe masked mean pooling to avoid numerical instability.
    """

    def __init__(self, config: Config):
        super(SiameseLSTM, self).__init__()

        self.embedding_dim = config.EMBEDDING_DIM
        self.vocab_size = config.VOCAB_SIZE
        self.hidden_dim = config.HIDDEN_DIM
        self.dropout_prob = config.DROPOUT
        self.num_classes = config.NUM_CLASSES

        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embedding_dim,
            padding_idx=0,
        )

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            batch_first=True,
            bidirectional=True,
        )

        # Input dimension: (Hidden * 2 directions) * 4 features + 3 length features
        # Features: Prompt, ResA, ResB, ResA - ResB, LenP, LenA, LenB
        input_dim = (self.hidden_dim * 2) * 4 + 3

        self.fc1 = nn.Linear(input_dim, self.hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.dropout_prob)
        self.fc2 = nn.Linear(self.hidden_dim, self.num_classes)

    def forward(self, prompt, response_a, response_b, lengths):
        # Encode sequences
        v_p = self._encode_sequence(prompt)
        v_a = self._encode_sequence(response_a)
        v_b = self._encode_sequence(response_b)

        # Difference vector
        v_diff = v_a - v_b

        # Concatenate embeddings and length features
        features = torch.cat([v_p, v_a, v_b, v_diff, lengths], dim=1)

        x = self.fc1(features)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x)

        return logits

    def _encode_sequence(self, x):
        # x: (batch, seq_len)
        embeds = self.embedding(x)  # (batch, seq, embed_dim)

        # LSTM forward
        # output: (batch, seq, hidden*2)
        output, _ = self.lstm(embeds)

        # Create mask for padding
        mask = (x != 0).float().unsqueeze(-1)  # (batch, seq, 1)

        # Masked Mean Pooling (Cite solution_lesson_node_00002)
        # Zero out padding positions in output
        masked_output = output * mask

        # Sum valid outputs
        sum_output = torch.sum(masked_output, dim=1)

        # Count valid tokens
        counts = torch.sum(mask, dim=1).clamp(min=1e-9)

        # Mean
        avg_output = sum_output / counts

        return avg_output
