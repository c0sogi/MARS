import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config


class HybridDualEncoder(nn.Module):
    """
    Hybrid-Depth Asymmetric Dual-Encoder for StackExchange QA Labeling.

    Architecture:
    - Question Branch: roberta-base (12 layers)
    - Answer Branch: distilroberta-base (6 layers)
    - Pooling: Masked Global Average and Max Pooling
    - Fusion: Interaction-Aware Fusion (Concat + Prod + Diff)
    - Head: Residual Projection Block
    """

    def __init__(self):
        super().__init__()

        # =====================================================================
        # 1. Asymmetric Backbones
        # =====================================================================
        # Question Branch: Deeper model for hierarchical structure (Title + Body)
        self.q_config = AutoConfig.from_pretrained(Config.QUESTION_MODEL_NAME)
        self.q_backbone = AutoModel.from_pretrained(
            Config.QUESTION_MODEL_NAME, config=self.q_config
        )

        # Answer Branch: Distilled model for efficiency
        self.a_config = AutoConfig.from_pretrained(Config.ANSWER_MODEL_NAME)
        self.a_backbone = AutoModel.from_pretrained(
            Config.ANSWER_MODEL_NAME, config=self.a_config
        )

        # Ensure compatibility
        if self.q_config.hidden_size != self.a_config.hidden_size:
            raise ValueError(
                f"Hidden size mismatch: Q ({self.q_config.hidden_size}) vs A ({self.a_config.hidden_size})"
            )

        self.hidden_size = self.q_config.hidden_size

        # =====================================================================
        # 2. Fusion Layer
        # =====================================================================
        # Components: [u_avg, u_max, v_avg, v_max, u_avg*v_avg, |u_avg-v_avg|]
        # Total dimension = 6 * hidden_size
        self.fusion_dim = 6 * self.hidden_size
        self.layer_norm = nn.LayerNorm(self.fusion_dim)

        # =====================================================================
        # 3. Residual Interaction Head
        # =====================================================================
        # Architecture: Output = Linear(Concat(F, Dropout(ReLU(Linear(F)))))
        self.inner_dim = 512  # Intermediate projection dimension

        # Path 1: Non-linear transformation
        self.head_proj_1 = nn.Linear(self.fusion_dim, self.inner_dim)
        self.head_act = nn.ReLU()
        self.head_drop = nn.Dropout(0.1)

        # Path 2: Final projection (takes Concat[F, Path1])
        self.head_proj_2 = nn.Linear(
            self.fusion_dim + self.inner_dim, Config.NUM_TARGETS
        )

        # Initialize head weights
        self._init_head_weights()

    def _init_head_weights(self):
        """Xavier initialization for the head layers."""
        for m in [self.head_proj_1, self.head_proj_2]:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _masked_pooling(self, hidden_state, attention_mask):
        """
        Applies Masked Global Average and Max Pooling.

        Args:
            hidden_state: (Batch, SeqLen, Hidden)
            attention_mask: (Batch, SeqLen)

        Returns:
            avg_pool: (Batch, Hidden)
            max_pool: (Batch, Hidden)
        """
        # Expand mask for broadcasting: (Batch, SeqLen, 1)
        mask_expanded = attention_mask.unsqueeze(-1).float()

        # --- Average Pooling ---
        # Sum hidden states of valid tokens
        sum_embeddings = torch.sum(hidden_state * mask_expanded, dim=1)
        # Count valid tokens (clamp to avoid division by zero)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        avg_pool = sum_embeddings / sum_mask

        # --- Max Pooling ---
        # Mask padding tokens with a large negative value so they aren't selected
        hidden_masked = hidden_state.clone()
        hidden_masked[attention_mask == 0] = -1e9
        max_pool = torch.max(hidden_masked, dim=1)[0]

        return avg_pool, max_pool

    def forward(self, input_ids_q, attention_mask_q, input_ids_a, attention_mask_a):
        """
        Forward pass of the Hybrid Dual Encoder.
        """
        # 1. Encode Question
        q_out = self.q_backbone(input_ids=input_ids_q, attention_mask=attention_mask_q)
        q_hidden = q_out.last_hidden_state  # (Batch, Len_Q, Hidden)

        # 2. Encode Answer
        a_out = self.a_backbone(input_ids=input_ids_a, attention_mask=attention_mask_a)
        a_hidden = a_out.last_hidden_state  # (Batch, Len_A, Hidden)

        # 3. Pooling
        u_avg, u_max = self._masked_pooling(q_hidden, attention_mask_q)
        v_avg, v_max = self._masked_pooling(a_hidden, attention_mask_a)

        # 4. Interaction-Aware Fusion
        # Compute explicit interactions on Average Pooled vectors
        prod = u_avg * v_avg
        diff = torch.abs(u_avg - v_avg)

        # Construct Fused Vector F
        F = torch.cat([u_avg, u_max, v_avg, v_max, prod, diff], dim=1)

        # Apply Layer Normalization
        F = self.layer_norm(F)

        # 5. Residual Interaction Head
        # Non-linear path
        h = self.head_proj_1(F)
        h = self.head_act(h)
        h = self.head_drop(h)

        # Skip connection: Concatenate original features F with transformed features h
        combined = torch.cat([F, h], dim=1)

        # Final prediction
        logits = self.head_proj_2(combined)

        return logits
