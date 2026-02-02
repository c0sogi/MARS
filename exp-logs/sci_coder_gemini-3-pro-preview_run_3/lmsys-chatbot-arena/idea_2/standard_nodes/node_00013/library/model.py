import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class AttentionPooling(nn.Module):
    """
    Learned Attention Pooling layer.
    Computes a weighted sum of hidden states where weights are learned.
    Cite solution_lesson_node_00011
    """

    def __init__(self, in_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.Tanh(),
            nn.Linear(in_dim, 1, bias=False),
        )

    def forward(self, x, mask):
        # x: (Batch, Length, Dim)
        # mask: (Batch, Length)

        # Compute attention scores
        w = self.attention(x)  # (Batch, Length, 1)
        scores = w.squeeze(-1)  # (Batch, Length)

        # Masking: Fill pad positions with large negative value
        # Cite solution_lesson_node_00002: Safe masking
        scores = scores.masked_fill(mask == 0, -1e9)

        # Softmax to get weights
        weights = torch.softmax(scores, dim=1)  # (Batch, Length)

        # Weighted sum
        # weights.unsqueeze(-1): (Batch, Length, 1)
        out = torch.sum(x * weights.unsqueeze(-1), dim=1)  # (Batch, Dim)

        return out


class SiameseAttentionModel(nn.Module):
    """
    Siamese Bi-LSTM with Attention Pooling and Hybrid Features.
    Replaces complex ESIM with robust Late Interaction (Cite solution_lesson_node_00010).
    """

    def __init__(self):
        super().__init__()
        self.vocab_size = Config.VOCAB_SIZE
        self.embed_dim = Config.EMBEDDING_DIM
        self.hidden_dim = Config.HIDDEN_DIM
        self.num_classes = Config.NUM_CLASSES
        self.dropout_rate = Config.DROPOUT

        # 1. Embedding Layer
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=0)

        # 2. Context Encoder (Shared Bi-LSTM)
        # Output dim will be 2 * hidden_dim due to bidirectional=True
        self.encoder = nn.LSTM(
            self.embed_dim, self.hidden_dim, batch_first=True, bidirectional=True
        )

        self.dropout = nn.Dropout(self.dropout_rate)

        # 3. Attention Pooling
        # Input to pooling is 2 * hidden_dim
        self.pooler = AttentionPooling(2 * self.hidden_dim)

        # 4. Classifier
        # Features:
        # - Pooled Prompt (2H)
        # - Pooled A (2H)
        # - Pooled B (2H)
        # - Diff |A - B| (2H)
        # - Prod A * B (2H)
        # - Scalars (3)
        # Total = 10 * hidden_dim + 3
        classifier_input_dim = (10 * self.hidden_dim) + 3

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, self.num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if "weight" in name:
                        nn.init.xavier_uniform_(param)
                    elif "bias" in name:
                        nn.init.zeros_(param)

    def forward(self, prompt_ids, res_a_ids, res_b_ids, scalars):
        # Create Masks
        mask_p = (prompt_ids != 0).float()
        mask_a = (res_a_ids != 0).float()
        mask_b = (res_b_ids != 0).float()

        # Embed
        emb_p = self.dropout(self.embedding(prompt_ids))
        emb_a = self.dropout(self.embedding(res_a_ids))
        emb_b = self.dropout(self.embedding(res_b_ids))

        # Encode (Shared LSTM)
        enc_p, _ = self.encoder(emb_p)
        enc_a, _ = self.encoder(emb_a)
        enc_b, _ = self.encoder(emb_b)

        # Apply dropout to encoded states
        enc_p = self.dropout(enc_p)
        enc_a = self.dropout(enc_a)
        enc_b = self.dropout(enc_b)

        # Pooling (Attention)
        pool_p = self.pooler(enc_p, mask_p)
        pool_a = self.pooler(enc_a, mask_a)
        pool_b = self.pooler(enc_b, mask_b)

        # Interaction Features
        diff = torch.abs(pool_a - pool_b)
        prod = pool_a * pool_b

        # Concatenate (Hybrid Input - Cite solution_lesson_node_00004)
        combined = torch.cat([pool_p, pool_a, pool_b, diff, prod, scalars], dim=1)

        # Classify
        logits = self.classifier(combined)

        return logits
