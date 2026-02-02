import torch
import torch.nn as nn
from library.config import Config


class AttentionPooling(nn.Module):
    """
    Learned Attention Pooling layer.
    Cite Lesson 00011: Superiority of Learned Attention Pooling.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False),
        )

    def forward(self, x, mask=None):
        # x: (batch, seq, hidden)
        w = self.attention(x)  # (batch, seq, 1)
        scores = w.squeeze(-1)  # (batch, seq)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        weights = torch.softmax(scores, dim=1)  # (batch, seq)
        out = torch.bmm(weights.unsqueeze(1), x).squeeze(1)
        return out


class SiameseBiLSTMAttention(nn.Module):
    """
    Siamese Bi-Directional LSTM with Attention Pooling and Hybrid Inputs.
    Cite Lesson 00015, 00008, 00004.
    """

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(
            Config.VOCAB_SIZE, Config.EMBED_DIM, padding_idx=0
        )

        self.lstm = nn.LSTM(
            Config.EMBED_DIM, Config.HIDDEN_DIM, batch_first=True, bidirectional=True
        )

        # Bidirectional output dim is hidden_dim * 2
        self.lstm_out_dim = Config.HIDDEN_DIM * 2
        self.attention = AttentionPooling(self.lstm_out_dim)

        # Features:
        # 3 embeddings (Prompt, A, B)
        # 2 interactions (|A-B|, A*B)
        # 3 scalars
        # Total = 5 * lstm_out_dim + 3
        self.classifier_input_dim = (5 * self.lstm_out_dim) + 3

        self.dropout = nn.Dropout(Config.HIDDEN_DROPOUT_PROB)

        self.classifier = nn.Sequential(
            nn.Linear(self.classifier_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(Config.HIDDEN_DROPOUT_PROB),
            nn.Linear(128, Config.NUM_LABELS),
        )

    def forward_one(self, input_ids):
        # Create mask (non-zero tokens)
        mask = (input_ids != 0).float()

        emb = self.embedding(input_ids)
        out, _ = self.lstm(emb)
        pooled = self.attention(out, mask)
        return pooled

    def forward(self, input_ids_p, input_ids_a, input_ids_b, scalar_features):
        # Encode all inputs using shared weights (Siamese)
        u_p = self.forward_one(input_ids_p)
        u_a = self.forward_one(input_ids_a)
        u_b = self.forward_one(input_ids_b)

        # Interaction Features (Cite Lesson 00001, 00008)
        diff = torch.abs(u_a - u_b)
        prod = u_a * u_b

        # Concatenate all features
        combined = torch.cat([u_p, u_a, u_b, diff, prod, scalar_features], dim=1)

        combined = self.dropout(combined)
        logits = self.classifier(combined)
        return logits
