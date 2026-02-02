import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class DistilRobertaDualEncoder(nn.Module):
    """
    DistilRoBERTa Dual-Encoder (Full Fine-Tuning).
    Cite Lesson 00068: Full Fine-Tuning of lightweight models outperforms LoRA on large models.
    """

    def __init__(
        self,
        model_name="distilroberta-base",
        num_labels=30,
    ):
        super().__init__()

        # 1. Load Backbone
        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.config.hidden_size

        # 3. Fusion & Head
        # Input dim: 4 original pools (u_avg, u_max, v_avg, v_max) + 2 interactions (prod, diff)
        # Total = 6 * hidden_size
        fusion_dim = self.hidden_size * 6

        # Cite Lesson 00054: Minimalist Projection Heads Preferable
        # Cite Lesson 00016: Monolithic Head
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, num_labels),
        )

        # Initialize head weights
        self._init_weights(self.head)

    def _init_weights(self, module):
        """Initialize weights for the custom head."""
        if isinstance(module, nn.Linear):
            # Cite Lesson 00051: Initialize to match backbone scale (Normal 0.02)
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Sequential):
            for sub_module in module:
                self._init_weights(sub_module)

    def _pool(self, last_hidden_state, attention_mask):
        """
        Applies Masked Global Average Pooling and Masked Max Pooling.
        """
        # Expand mask: (batch, seq_len) -> (batch, seq_len, hidden_size)
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )

        # Average Pooling
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        avg_pool = sum_embeddings / sum_mask

        # Max Pooling
        # Mask out padding tokens with a very large negative value
        hidden_masked = last_hidden_state.clone()
        hidden_masked = hidden_masked.masked_fill(input_mask_expanded == 0, -1e9)
        max_pool = torch.max(hidden_masked, 1)[0]

        return avg_pool, max_pool

    def forward(
        self,
        q_input_ids,
        q_attention_mask,
        a_input_ids,
        a_attention_mask,
        labels=None,
        **kwargs
    ):
        # 1. Process Question Branch
        q_outputs = self.backbone(
            input_ids=q_input_ids, attention_mask=q_attention_mask
        )
        q_hidden = q_outputs.last_hidden_state
        u_avg, u_max = self._pool(q_hidden, q_attention_mask)

        # 2. Process Answer Branch
        a_outputs = self.backbone(
            input_ids=a_input_ids, attention_mask=a_attention_mask
        )
        a_hidden = a_outputs.last_hidden_state
        v_avg, v_max = self._pool(a_hidden, a_attention_mask)

        # 3. Interaction Features
        # Cite Lesson 00057: Simplicity in Feature Fusion
        prod = u_avg * v_avg
        diff = torch.abs(u_avg - v_avg)

        # 4. Concatenation
        fused_vector = torch.cat([u_avg, u_max, v_avg, v_max, prod, diff], dim=1)

        # 5. Prediction Head
        logits = self.head(fused_vector)

        return logits
