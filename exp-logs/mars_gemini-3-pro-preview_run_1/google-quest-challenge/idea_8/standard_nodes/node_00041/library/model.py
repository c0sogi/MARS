import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from library.config import config


class MeanMaxPooling(nn.Module):
    """
    Performs Mean and Max pooling on the last hidden state of a transformer model.
    Handles attention masks to ignore padding tokens.
    """

    def __init__(self):
        super(MeanMaxPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        # last_hidden_state: (batch_size, seq_len, hidden_size)
        # attention_mask: (batch_size, seq_len)

        # Expand mask to match hidden state dimensions
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )

        # Sum of embeddings for mean pooling
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)  # Avoid division by zero
        mean_embeddings = sum_embeddings / sum_mask

        # Max pooling
        # Set padding tokens to a very small number so they are not selected as max
        last_hidden_state[input_mask_expanded == 0] = -1e9
        max_embeddings = torch.max(last_hidden_state, 1)[0]

        return mean_embeddings, max_embeddings


class DebertaDualEncoder(nn.Module):
    """
    DistilRoBERTa Dual-Encoder.
    Fuses text features from Question and Answer branches (Avg, Max, Prod, Diff).
    """

    def __init__(self, meta_dims):
        """
        Args:
            meta_dims (dict): Unused, kept for interface compatibility.
        """
        super(DebertaDualEncoder, self).__init__()

        # 1. Backbone
        self.config = AutoConfig.from_pretrained(config.model_name)
        self.backbone = AutoModel.from_pretrained(config.model_name, config=self.config)

        self.hidden_size = self.config.hidden_size

        # 2. Pooling
        self.pooler = MeanMaxPooling()

        # 3. Fusion Dimension Calculation
        # Text: Q_avg, Q_max, A_avg, A_max, Q_avg*A_avg, |Q_avg-A_avg| -> 6 vectors
        total_fusion_dim = self.hidden_size * 6

        # 4. Fusion Normalization
        self.fusion_norm = nn.LayerNorm(total_fusion_dim)

        # 5. Non-Linear MLP Head
        # Structure: Linear -> GELU -> Dropout -> Linear (logits)
        self.inter_dim = self.hidden_size  # Intermediate dimension

        self.head_fc1 = nn.Linear(total_fusion_dim, self.inter_dim)
        self.head_act = nn.GELU()
        self.head_norm = nn.LayerNorm(self.inter_dim)

        # Multi-Sample Dropout
        self.dropouts = nn.ModuleList([nn.Dropout(0.5) for _ in range(5)])

        # Final projection to 30 targets
        self.head_fc2 = nn.Linear(self.inter_dim, len(config.target_cols))

        # Initialize weights for new layers
        self._init_weights(self.fusion_norm)
        self._init_weights(self.head_fc1)
        self._init_weights(self.head_norm)
        self._init_weights(self.head_fc2)

    def _init_weights(self, module):
        """Initialize weights for new layers similar to transformers."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
        self,
        q_input_ids,
        q_attention_mask,
        a_input_ids,
        a_attention_mask,
    ):
        # --- Branch 1: Question ---
        q_out = self.backbone(input_ids=q_input_ids, attention_mask=q_attention_mask)
        q_avg, q_max = self.pooler(q_out.last_hidden_state, q_attention_mask)

        # --- Branch 2: Answer ---
        a_out = self.backbone(input_ids=a_input_ids, attention_mask=a_attention_mask)
        a_avg, a_max = self.pooler(a_out.last_hidden_state, a_attention_mask)

        # --- Interactions (on Avg Pooled features) ---
        # Element-wise product
        inter_prod = q_avg * a_avg
        # Absolute difference
        inter_diff = torch.abs(q_avg - a_avg)

        # --- Fusion ---
        # Concatenate all features
        # [Q_avg, Q_max, A_avg, A_max, Prod, Diff]
        fused_features = torch.cat(
            [q_avg, q_max, a_avg, a_max, inter_prod, inter_diff],
            dim=1,
        )

        # Normalize fused vector
        fused_features = self.fusion_norm(fused_features)

        # --- MLP Head ---
        x = self.head_fc1(fused_features)
        x = self.head_act(x)
        x = self.head_norm(x)

        # Multi-Sample Dropout
        # Average the logits from multiple dropout masks
        logits = torch.mean(
            torch.stack(
                [self.head_fc2(dropout(x)) for dropout in self.dropouts], dim=0
            ),
            dim=0,
        )

        return logits
