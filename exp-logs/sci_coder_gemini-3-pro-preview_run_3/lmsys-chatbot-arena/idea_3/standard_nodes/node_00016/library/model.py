import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class AttentionPooling(nn.Module):
    """
    Cite solution_lesson_node_00011: "Superiority of Learned Attention Pooling"
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1)
        )

    def forward(self, x, mask=None):
        # x: (batch, seq_len, hidden_size)
        # mask: (batch, seq_len) - 0 for pad, 1 for real

        # Calculate weights
        weights = self.attention(x)  # (batch, seq_len, 1)

        if mask is not None:
            # Mask padding
            weights = weights.masked_fill(mask.unsqueeze(-1) == 0, -1e9)

        weights = F.softmax(weights, dim=1)

        # Weighted sum
        # (batch, seq_len, 1) * (batch, seq_len, hidden_size) -> sum over seq_len
        pooled = torch.sum(weights * x, dim=1)
        return pooled


class SiameseBiLSTMAttention(nn.Module):
    """
    Siamese Bi-LSTM with Attention Pooling and Hybrid Inputs.
    Cite solution_lesson_node_00008: "Siamese Architectures Outperform Cross-Encoders"
    """

    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(
            Config.VOCAB_SIZE, Config.EMBED_DIM, padding_idx=0
        )

        self.lstm = nn.LSTM(
            input_size=Config.EMBED_DIM,
            hidden_size=Config.HIDDEN_DIM,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # Bidirectional -> 2 * hidden_dim
        self.pooler = AttentionPooling(Config.HIDDEN_DIM * 2)

        # Input features:
        # Prompt Embedding (2*H)
        # Resp A Embedding (2*H)
        # Resp B Embedding (2*H)
        # Diff (2*H)
        # Scalars (3)
        # Total = 8*H + 3
        self.combined_dim = (Config.HIDDEN_DIM * 2 * 4) + 3

        self.dropout = nn.Dropout(Config.HIDDEN_DROPOUT_PROB)
        self.classifier = nn.Sequential(
            nn.Linear(self.combined_dim, 128),
            nn.ReLU(),
            nn.Dropout(Config.HIDDEN_DROPOUT_PROB),
            nn.Linear(128, Config.NUM_LABELS),
        )

    def forward_one(self, ids):
        # Create mask (non-zero tokens)
        mask = (ids != 0).long()

        embeds = self.embedding(ids)
        out, _ = self.lstm(embeds)
        pooled = self.pooler(out, mask)
        return pooled

    def forward(self, ids_prompt, ids_a, ids_b, scalar_features):
        # Encode each sequence
        # Cite solution_lesson_node_00010: "Late Interaction"
        u = self.forward_one(ids_prompt)
        v = self.forward_one(ids_a)
        w = self.forward_one(ids_b)

        # Interaction features
        diff = torch.abs(v - w)

        # Concatenate
        combined = torch.cat([u, v, w, diff, scalar_features], dim=1)
        combined = self.dropout(combined)

        logits = self.classifier(combined)
        return logits
