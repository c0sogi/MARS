import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from peft import LoraConfig, get_peft_model, TaskType


class LoRADebertaDualEncoder(nn.Module):
    """
    LoRA-Augmented DeBERTa-v3-Base Dual-Encoder.

    Architecture:
    1. Shared Backbone: microsoft/deberta-v3-base with LoRA adapters.
    2. Dual-Branch: Independent processing of Question and Answer inputs.
    3. Pooling: Masked Global Average + Masked Max Pooling.
    4. Fusion: Concatenation of [u_avg, u_max, v_avg, v_max, u_avg*v_avg, |u_avg-v_avg|].
    5. Head: LayerNorm -> Linear -> ReLU -> Dropout -> Linear (30 targets).
    """

    def __init__(
        self,
        model_name="microsoft/deberta-v3-base",
        num_labels=30,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.1,
    ):
        super().__init__()

        # 1. Load Backbone
        self.config = AutoConfig.from_pretrained(model_name)
        base_model = AutoModel.from_pretrained(model_name)

        # 2. Configure LoRA
        # Target modules: Query, Key, Value, and Output projections
        # In DeBERTa, these are typically query_proj, key_proj, value_proj, and output.dense
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["query_proj", "key_proj", "value_proj", "output.dense"],
            bias="none",
        )

        # Wrap model with PEFT
        self.backbone = get_peft_model(base_model, peft_config)
        self.hidden_size = self.config.hidden_size

        # 3. Fusion & Head
        # Input dim: 4 original pools (u_avg, u_max, v_avg, v_max) + 2 interactions (prod, diff)
        # Total = 6 * hidden_size
        fusion_dim = self.hidden_size * 6

        self.head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
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
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
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
        # Clone to avoid in-place modification errors
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

        # 3. Interaction Features (on Average Pooled vectors)
        # Element-wise product
        prod = u_avg * v_avg
        # Absolute difference
        diff = torch.abs(u_avg - v_avg)

        # 4. Concatenation
        # [u_avg, u_max, v_avg, v_max, prod, diff]
        fused_vector = torch.cat([u_avg, u_max, v_avg, v_max, prod, diff], dim=1)

        # 5. Prediction Head
        logits = self.head(fused_vector)

        return logits
