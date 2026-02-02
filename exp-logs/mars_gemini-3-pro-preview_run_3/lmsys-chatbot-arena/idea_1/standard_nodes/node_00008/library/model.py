import torch
import torch.nn as nn
from library.config import Config


class CrossEncoderLSTM(nn.Module):
    """
    Cross-Encoder Bi-Directional LSTM for Chatbot Arena preference prediction.
    Cite solution_lesson_node_00006: Replaces Siamese architecture with Cross-Encoder to capture interactions.
    Cite solution_lesson_node_00004: Uses Hybrid Inputs (Lengths).
    """

    def __init__(self, config: Config):
        super(CrossEncoderLSTM, self).__init__()

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

        # Input dimension: (Hidden * 2 directions) + 3 length features
        input_dim = (self.hidden_dim * 2) + 3

        self.fc1 = nn.Linear(input_dim, self.hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.dropout_prob)
        self.fc2 = nn.Linear(self.hidden_dim, self.num_classes)

    def forward(self, input_ids, lengths):
        # input_ids: (batch, seq_len) - Concatenated P+SEP+A+SEP+B
        embeds = self.embedding(input_ids)  # (batch, seq, embed_dim)

        # LSTM forward
        # output: (batch, seq, hidden*2)
        output, _ = self.lstm(embeds)

        # Global Max Pooling (often better for classification than Mean)
        # Cite solution_lesson_node_00002: Safe masking
        mask = (input_ids != 0).float().unsqueeze(-1)  # (batch, seq, 1)

        # Mask padded values with large negative number
        # We ensure we don't mask everything by checking content later, though P+A+B is never empty
        masked_output = output * mask + (1 - mask) * -1e9

        # Max pooling
        pooled, _ = torch.max(masked_output, dim=1)

        # Concatenate embeddings and length features
        features = torch.cat([pooled, lengths], dim=1)

        x = self.fc1(features)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x)

        return logits
