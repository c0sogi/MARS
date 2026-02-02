import torch
import torch.nn as nn
from transformers import AutoModel
from library.config import Config


class TripleBranchDistilRoBERTa(nn.Module):
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
        #   - Title: Avg, Max (2)
        #   - Body:  Avg, Max (2)
        #   - Answer: Avg, Max (2)
        #   -> Subtotal: 6 vectors
        #
        # Interaction Features (computed on Avg vectors only):
        #   - Title-Answer: Product, AbsDiff (2)
        #   - Body-Answer:  Product, AbsDiff (2)
        #   - Title-Body:   Product, AbsDiff (2)
        #   -> Subtotal: 6 vectors
        #
        # Total Vectors: 12
        self.fusion_dim = 12 * self.hidden_size

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
        title_input_ids,
        title_attention_mask,
        body_input_ids,
        body_attention_mask,
        answer_input_ids,
        answer_attention_mask,
    ):
        """
        Processes the three input streams, fuses them, and predicts targets.
        Arguments match the keys produced by the Collate class in dataset.py.
        """

        # --- Branch 1: Title ---
        t_out = self.backbone(
            input_ids=title_input_ids, attention_mask=title_attention_mask
        )
        t_avg, t_max = self._pool(t_out.last_hidden_state, title_attention_mask)

        # --- Branch 2: Body ---
        b_out = self.backbone(
            input_ids=body_input_ids, attention_mask=body_attention_mask
        )
        b_avg, b_max = self._pool(b_out.last_hidden_state, body_attention_mask)

        # --- Branch 3: Answer ---
        a_out = self.backbone(
            input_ids=answer_input_ids, attention_mask=answer_attention_mask
        )
        a_avg, a_max = self._pool(a_out.last_hidden_state, answer_attention_mask)

        # --- Granular Interaction Fusion (On Average Vectors) ---

        # 1. Title-Answer Interactions (Intent alignment)
        ta_prod = t_avg * a_avg
        ta_diff = torch.abs(t_avg - a_avg)

        # 2. Body-Answer Interactions (Detail coverage)
        ba_prod = b_avg * a_avg
        ba_diff = torch.abs(b_avg - a_avg)

        # 3. Title-Body Interactions (Consistency)
        tb_prod = t_avg * b_avg
        tb_diff = torch.abs(t_avg - b_avg)

        # --- Concatenation ---
        # Order: Raw Features (T, B, A) -> Interactions
        fused = torch.cat(
            [
                t_avg,
                t_max,
                b_avg,
                b_max,
                a_avg,
                a_max,
                ta_prod,
                ta_diff,
                ba_prod,
                ba_diff,
                tb_prod,
                tb_diff,
            ],
            dim=1,
        )

        # --- Normalization ---
        normed = self.layer_norm(fused)

        # --- Prediction ---
        logits = self.head(normed)

        return logits
