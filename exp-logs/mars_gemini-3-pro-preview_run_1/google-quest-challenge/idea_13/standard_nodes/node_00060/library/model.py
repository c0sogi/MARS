import torch
import torch.nn as nn
from transformers import AutoModel
from library.config import Config


class DualBranchDistilRoBERTa(nn.Module):
    def __init__(self):
        super().__init__()

        # 1. Backbone: Shared DistilRoBERTa
        self.backbone = AutoModel.from_pretrained(Config.MODEL_NAME)
        self.hidden_size = (
            self.backbone.config.hidden_size
        )  # 768 for distilroberta-base

        # 2. Fusion Dimension Calculation
        # We concatenate the following vectors (all size H):
        # Raw Pooled Features:
        #   - Question: Avg, Max (2)
        #   - Answer:   Avg, Max (2)
        #   -> Subtotal: 4 vectors
        #
        # Interaction Features (computed on Avg vectors only):
        # Cite solution_lesson_node_00007: Interaction Heuristics on Centroids
        #   - Question-Answer: Product, AbsDiff (2)
        #   -> Subtotal: 2 vectors
        #
        # Total Vectors: 6
        self.fusion_dim = 6 * self.hidden_size

        # 3. Normalization
        self.layer_norm = nn.LayerNorm(self.fusion_dim)

        # 4. Minimalist Monolithic Head
        # Linear -> ReLU -> Dropout -> Linear
        self.head = nn.Sequential(
            nn.Linear(self.fusion_dim, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, Config.NUM_LABELS),
        )

        # 5. Initialization
        self._init_head()

    def _init_head(self):
        """
        Initialize the head's weights using a Normal distribution (mu=0, sigma=0.02)
        to match the backbone's scale.
        """
        for module in self.head:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _pool(self, last_hidden_state, attention_mask):
        """
        Applies Masked Global Average and Max Pooling.

        Args:
            last_hidden_state: (Batch, SeqLen, Hidden)
            attention_mask: (Batch, SeqLen)

        Returns:
            avg_pool: (Batch, Hidden)
            max_pool: (Batch, Hidden)
        """
        # Expand mask to (Batch, SeqLen, 1) for broadcasting
        mask_expanded = attention_mask.unsqueeze(-1).float()

        # --- Average Pooling ---
        # Sum hidden states where mask is 1
        sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
        # Sum mask (count of valid tokens)
        sum_mask = mask_expanded.sum(dim=1)
        # Clamp to avoid division by zero
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        avg_pool = sum_embeddings / sum_mask

        # --- Max Pooling ---
        # We need to ignore padding tokens in max calculation.
        # Set padding positions to a very large negative number.
        # Note: We use -1e9 instead of -inf to be safe with fp16/gradients.
        neg_inf = -1e9
        masked_hidden_state = last_hidden_state.clone()

        # Create boolean mask where True indicates padding (0 in attention_mask)
        is_padding = mask_expanded == 0

        # Apply mask
        masked_hidden_state[is_padding.expand_as(masked_hidden_state)] = neg_inf

        # Max over sequence length dimension
        max_pool = torch.max(masked_hidden_state, dim=1)[0]

        return avg_pool, max_pool

    def forward(
        self,
        question_input_ids,
        question_attention_mask,
        answer_input_ids,
        answer_attention_mask,
    ):
        """
        Processes the two input streams, fuses them, and predicts targets.
        Arguments match the keys produced by the Collate class in dataset.py.
        """

        # --- Branch 1: Question ---
        q_out = self.backbone(
            input_ids=question_input_ids, attention_mask=question_attention_mask
        )
        q_avg, q_max = self._pool(q_out.last_hidden_state, question_attention_mask)

        # --- Branch 2: Answer ---
        a_out = self.backbone(
            input_ids=answer_input_ids, attention_mask=answer_attention_mask
        )
        a_avg, a_max = self._pool(a_out.last_hidden_state, answer_attention_mask)

        # --- Interaction Fusion (On Average Vectors) ---
        # Cite solution_lesson_node_00007: Interactions on Avg only
        qa_prod = q_avg * a_avg
        qa_diff = torch.abs(q_avg - a_avg)

        # --- Concatenation ---
        # Order: Raw Features (Q, A) -> Interactions
        fused = torch.cat(
            [
                q_avg,
                q_max,
                a_avg,
                a_max,
                qa_prod,
                qa_diff,
            ],
            dim=1,
        )

        # --- Normalization ---
        normed = self.layer_norm(fused)

        # --- Prediction ---
        logits = self.head(normed)

        return logits
