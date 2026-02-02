import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class ESIMHybridModel(nn.Module):
    """
    Siamese Bi-LSTM with Cross-Attention (ESIM-inspired) and Hybrid Features.

    Architecture:
    1. Shared Embedding
    2. Shared Bi-LSTM Encoder
    3. Cross-Attention between Response A and Response B
    4. Enhancement Layer (Difference & Product)
    5. Projection Layer
    6. Global Avg & Max Pooling (masked)
    7. Concatenation with Prompt encoding and Scalar features
    8. MLP Classifier
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

        # 3. Projection Layer (for Enhanced representations)
        # Input: [h; align; h-align; h*align] where h is 2*hidden_dim (BiLSTM output)
        # Total input dim = 4 * (2 * hidden_dim) = 8 * hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(8 * self.hidden_dim, self.hidden_dim), nn.ReLU()
        )

        # 4. Classifier
        # Inputs to classifier:
        # - Pooled A: Avg(proj) + Max(proj) -> 2 * hidden_dim
        # - Pooled B: Avg(proj) + Max(proj) -> 2 * hidden_dim
        # - Pooled Prompt: Avg(enc) + Max(enc) -> 2 * (2*hidden_dim) = 4 * hidden_dim
        # - Scalars: 3 features

        classifier_input_dim = (
            (2 * self.hidden_dim) + (2 * self.hidden_dim) + (4 * self.hidden_dim) + 3
        )

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
        Xavier initialization for Linear and LSTM layers.
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
        # Ensure masks are on the same device as inputs
        mask_p = (prompt_ids != 0).float()
        mask_a = (res_a_ids != 0).float()
        mask_b = (res_b_ids != 0).float()

        # --- 1. Embedding & Encoding ---
        emb_p = self.embedding(prompt_ids)
        emb_a = self.embedding(res_a_ids)
        emb_b = self.embedding(res_b_ids)

        # Apply dropout to embeddings
        emb_p = self.dropout(emb_p)
        emb_a = self.dropout(emb_a)
        emb_b = self.dropout(emb_b)

        # Encode sequences
        # enc outputs: (Batch, Length, 2*Hidden)
        enc_p, _ = self.encoder(emb_p)
        enc_a, _ = self.encoder(emb_a)
        enc_b, _ = self.encoder(emb_b)

        # Apply dropout to encoded states
        enc_p = self.dropout(enc_p)
        enc_a = self.dropout(enc_a)
        enc_b = self.dropout(enc_b)

        # --- 2. Cross-Attention (Interaction) ---
        # Compute Attention Matrix E = A * B^T
        # Dimensions: (B, La, 2H) * (B, 2H, Lb) -> (B, La, Lb)
        attention = torch.matmul(enc_a, enc_b.transpose(1, 2))

        # Masking Attention
        # Mask out positions where either A or B is padding
        # mask_a: (B, La), mask_b: (B, Lb)
        mask_attn = mask_a.unsqueeze(2) * mask_b.unsqueeze(1)  # (B, La, Lb)

        # Fill invalid positions with very small number before softmax
        attention = attention.masked_fill(mask_attn == 0, -1e9)

        # Softmax to get alignment weights
        prob_a = F.softmax(attention, dim=2)  # (B, La, Lb) - Align A to B (sum over B)
        prob_b = F.softmax(attention, dim=1)  # (B, La, Lb) - Align B to A (sum over A)

        # Compute Aligned Representations
        # aligned_a: For each token in A, weighted sum of B's states
        aligned_a = torch.matmul(prob_a, enc_b)  # (B, La, 2H)

        # aligned_b: For each token in B, weighted sum of A's states
        # Need to transpose prob_b to (B, Lb, La) to multiply with enc_a (B, La, 2H)
        aligned_b = torch.matmul(prob_b.transpose(1, 2), enc_a)  # (B, Lb, 2H)

        # --- 3. Enhancement ---
        # Concatenate: [Original; Aligned; Difference; Product]
        enhanced_a = torch.cat(
            [enc_a, aligned_a, enc_a - aligned_a, enc_a * aligned_a], dim=-1
        )
        enhanced_b = torch.cat(
            [enc_b, aligned_b, enc_b - aligned_b, enc_b * aligned_b], dim=-1
        )

        # --- 4. Projection ---
        # Reduce dimensionality before pooling
        proj_a = self.projection(enhanced_a)  # (B, La, H)
        proj_b = self.projection(enhanced_b)  # (B, Lb, H)

        # --- 5. Pooling ---
        def apply_pooling(tensor, mask):
            """
            Applies Global Average and Max Pooling with proper masking.
            tensor: (Batch, Length, Dim)
            mask: (Batch, Length)
            """
            mask_expanded = mask.unsqueeze(-1)  # (B, L, 1)

            # Avg Pooling
            # Zero out pad vectors, sum, then divide by valid length
            sum_pooled = torch.sum(tensor * mask_expanded, dim=1)
            lens = mask_expanded.sum(dim=1).clamp(min=1e-9)
            avg_pooled = sum_pooled / lens

            # Max Pooling
            # Fill pads with -1e9 so they are not selected as max
            tensor_masked = tensor.masked_fill(mask_expanded == 0, -1e9)
            max_pooled = torch.max(tensor_masked, dim=1)[0]

            return torch.cat([avg_pooled, max_pooled], dim=1)

        pooled_a = apply_pooling(proj_a, mask_a)  # (B, 2H)
        pooled_b = apply_pooling(proj_b, mask_b)  # (B, 2H)
        pooled_p = apply_pooling(enc_p, mask_p)  # (B, 4H) - enc_p was 2H

        # --- 6. Classification ---
        # Concatenate all features
        combined = torch.cat([pooled_a, pooled_b, pooled_p, scalars], dim=1)

        # Forward pass through MLP
        logits = self.classifier(combined)

        return logits
