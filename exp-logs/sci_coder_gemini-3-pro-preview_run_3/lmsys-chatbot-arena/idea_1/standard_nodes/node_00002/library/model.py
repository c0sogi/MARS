import torch
import torch.nn as nn
from library.config import Config


class SiameseLSTM(nn.Module):
    """
    Siamese Bi-Directional LSTM for Chatbot Arena preference prediction.
    Cite solution_lesson_node_00001: Replaces Global Pooling with Sequence-Aware LSTM.

    Architecture:
    1. Shared Embedding Layer.
    2. Shared Bi-LSTM Encoder: Processes Prompt, ResA, and ResB.
    3. Global Max Pooling: Extracts most salient features over time.
    4. Feature Concatenation: Combines [Prompt, ResA, ResB, ResA - ResB].
    5. MLP Classifier.
    """

    def __init__(self, config: Config):
        super(SiameseLSTM, self).__init__()

        self.vocab_size = config.VOCAB_SIZE
        self.embedding_dim = config.EMBEDDING_DIM
        self.hidden_dim = config.HIDDEN_DIM
        self.dropout_prob = config.DROPOUT
        self.num_classes = config.NUM_CLASSES

        # Shared Embedding Layer
        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embedding_dim,
            padding_idx=0,
        )

        # Shared Bi-Directional LSTM
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # Output dimension of Bi-LSTM is hidden_dim * 2
        self.lstm_out_dim = self.hidden_dim * 2

        # MLP Layers
        # Input: Prompt + ResA + ResB + (ResA - ResB) -> 4 * lstm_out_dim
        input_dim = self.lstm_out_dim * 4

        self.fc1 = nn.Linear(input_dim, self.hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.dropout_prob)
        self.fc2 = nn.Linear(self.hidden_dim, self.num_classes)

    def forward_encoder(self, x):
        """
        Encodes a sequence using Embedding -> Bi-LSTM -> Max Pooling.
        """
        # Create mask for non-padding tokens (padding_idx is 0)
        # Shape: (batch_size, seq_len, 1)
        mask = (x != 0).float().unsqueeze(-1)

        # Embeddings: (batch_size, seq_len, embedding_dim)
        embeds = self.embedding(x)

        # LSTM Forward
        # out: (batch_size, seq_len, lstm_out_dim)
        out, _ = self.lstm(embeds)

        # Masking for Max Pooling
        # We replace padded positions with a large negative number so they aren't selected
        # (1 - mask) is 1 where padding exists.
        out_masked = out * mask + (1 - mask) * -1e9

        # Global Max Pooling over sequence dimension
        # pooled: (batch_size, lstm_out_dim)
        pooled, _ = torch.max(out_masked, dim=1)

        return pooled

    def forward(self, prompt, response_a, response_b):
        # Encode inputs
        v_p = self.forward_encoder(prompt)
        v_a = self.forward_encoder(response_a)
        v_b = self.forward_encoder(response_b)

        # Comparison Feature
        v_diff = v_a - v_b

        # Concatenate
        features = torch.cat([v_p, v_a, v_b, v_diff], dim=1)

        # Classifier
        x = self.fc1(features)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x)

        return logits
