import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class SiameseHybridModel(nn.Module):
    """
    Siamese Bi-LSTM with Late Interaction and Hybrid Features.
    Cite Lesson 00010: Prioritizing Late Interaction over Cross-Attention.
    Cite Lesson 00004: Using Hybrid Inputs (scalars).
    Cite Lesson 00002: Safe Pooling to avoid numerical instability.
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

        # 3. Classifier
        # Inputs to classifier:
        # - Pooled Prompt (Avg+Max): 2 * (2*H) = 4H
        # - Pooled A (Avg+Max): 4H
        # - Pooled B (Avg+Max): 4H
        # - Diff (A-B): 4H
        # - Prod (A*B): 4H
        # - Scalars: 3
        # Total = 20 * H + 3
        # With H=128, 20*128 = 2560 + 3 = 2563

        classifier_input_dim = (5 * 4 * self.hidden_dim) + 3

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, self.num_classes),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """
        Xavier initialization.
        """
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
        # Create Masks (1 for content, 0 for pad)
        mask_p = (prompt_ids != 0).float()
        mask_a = (res_a_ids != 0).float()
        mask_b = (res_b_ids != 0).float()

        # --- 1. Embedding ---
        emb_p = self.dropout(self.embedding(prompt_ids))
        emb_a = self.dropout(self.embedding(res_a_ids))
        emb_b = self.dropout(self.embedding(res_b_ids))

        # --- 2. Encoding ---
        enc_p, _ = self.encoder(emb_p)
        enc_a, _ = self.encoder(emb_a)
        enc_b, _ = self.encoder(emb_b)

        enc_p = self.dropout(enc_p)
        enc_a = self.dropout(enc_a)
        enc_b = self.dropout(enc_b)

        # --- 3. Safe Pooling (Cite Lesson 00002) ---
        def apply_pooling(tensor, mask):
            mask_expanded = mask.unsqueeze(-1)  # (B, L, 1)

            # Avg Pooling
            sum_pooled = torch.sum(tensor * mask_expanded, dim=1)
            lens = mask_expanded.sum(dim=1).clamp(min=1e-9)
            avg_pooled = sum_pooled / lens

            # Max Pooling (Safe)
            # 1. Mask out padding with large negative
            tensor_masked = tensor.masked_fill(mask_expanded == 0, -1e9)
            # 2. Perform max
            max_pooled = torch.max(tensor_masked, dim=1)[0]
            # 3. Guard against empty sequences (all padding)
            # If sequence is empty, max_pooled will be -1e9. We zero it out.
            has_content = (mask.sum(dim=1) > 0).float().unsqueeze(-1)
            max_pooled = max_pooled * has_content

            return torch.cat([avg_pooled, max_pooled], dim=1)

        # Each pooled vector is 4 * Hidden (2H for Avg + 2H for Max)
        pooled_p = apply_pooling(enc_p, mask_p)
        pooled_a = apply_pooling(enc_a, mask_a)
        pooled_b = apply_pooling(enc_b, mask_b)

        # --- 4. Late Interaction (Cite Lesson 00010) ---
        diff_ab = pooled_a - pooled_b
        prod_ab = pooled_a * pooled_b

        # --- 5. Classification ---
        combined = torch.cat(
            [pooled_p, pooled_a, pooled_b, diff_ab, prod_ab, scalars], dim=1
        )

        logits = self.classifier(combined)

        return logits
