import torch
import torch.nn as nn
from library.config import Config


class AttentionPooling(nn.Module):
    """
    Attention-based pooling layer to aggregate sequence outputs.
    Cite solution_lesson_node_00001: Provides a better alternative to Global Average Pooling.
    """

    def __init__(self, input_dim):
        super(AttentionPooling, self).__init__()
        self.attention_weights = nn.Linear(input_dim, 1)

    def forward(self, x, mask):
        # x: (batch, seq, hidden)
        # mask: (batch, seq, 1)

        # Calculate attention scores
        scores = self.attention_weights(x)  # (batch, seq, 1)

        # Mask padding (Cite solution_lesson_node_00002: Handle masking safely)
        scores = scores.masked_fill(mask == 0, -1e9)

        # Softmax to get weights
        weights = torch.softmax(scores, dim=1)

        # Weighted sum
        context = torch.sum(x * weights, dim=1)
        return context


class SiameseLSTM(nn.Module):
    """
    Siamese Bi-Directional LSTM for Chatbot Arena preference prediction.
    Cite solution_lesson_node_00001: Replaces GAP with sequence-aware LSTM and Attention.
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

        # Attention Pooling
        self.attention = AttentionPooling(self.hidden_dim * 2)

        # Input dimension: (Hidden * 2 directions) * 5 features
        # Features: Prompt, ResA, ResB, ResA - ResB, ResA * ResB
        # Plus 3 scalar features (log lengths) (Cite solution_lesson_node_00004)
        input_dim = (self.hidden_dim * 2) * 5 + 3

        self.fc1 = nn.Linear(input_dim, self.hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.dropout_prob)
        self.fc2 = nn.Linear(self.hidden_dim, self.num_classes)

    def forward(self, prompt, response_a, response_b, lengths):
        # Encode sequences
        v_p = self._encode_sequence(prompt)
        v_a = self._encode_sequence(response_a)
        v_b = self._encode_sequence(response_b)

        # Interaction vectors
        v_diff = v_a - v_b
        v_prod = v_a * v_b

        # Concatenate embeddings and scalar features
        features = torch.cat([v_p, v_a, v_b, v_diff, v_prod, lengths], dim=1)

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

        # Attention Pooling
        pooled = self.attention(output, mask)

        return pooled
