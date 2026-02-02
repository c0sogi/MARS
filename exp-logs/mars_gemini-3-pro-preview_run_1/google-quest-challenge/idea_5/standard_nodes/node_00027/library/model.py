import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config


class QuestModel(nn.Module):
    """
    Lightweight DeBERTa-v3 Dual-Encoder with Multi-Sample Dropout.

    Architecture:
    1. Backbone: microsoft/deberta-v3-small (Shared weights, dual pass).
    2. Pooling: Masked Global Average and Max Pooling.
    3. Interaction: Element-wise Product and Abs Difference on Avg vectors.
    4. Fusion: Concatenation of [Avg_Q, Avg_A, Max_Q, Max_A, Prod, Diff].
    5. Head: LayerNorm -> Multi-Sample Dropout -> Linear.
    """

    def __init__(self):
        super().__init__()

        # Load Configuration
        self.config = AutoConfig.from_pretrained(Config.model_name)

        # Load Backbone
        # We use a single backbone instance for the Siamese/Dual-Encoder structure
        self.backbone = AutoModel.from_pretrained(Config.model_name, config=self.config)

        # Feature Dimensions
        self.hidden_size = self.config.hidden_size

        # Fusion Dimension Calculation:
        # 1. Q_avg (hidden_size)
        # 2. A_avg (hidden_size)
        # 3. Q_max (hidden_size)
        # 4. A_max (hidden_size)
        # 5. Interaction Product (hidden_size)
        # 6. Interaction Diff (hidden_size)
        # 7. Cosine Similarity (1)
        self.fusion_size = self.hidden_size * 6 + 1

        # Normalization layer applied to fused vector
        self.layer_norm = nn.LayerNorm(self.fusion_size)

        # Multi-Sample Dropout Head
        # Create K dropout instances
        self.dropouts = nn.ModuleList(
            [
                nn.Dropout(Config.hidden_dropout_prob)
                for _ in range(Config.n_dropout_samples)
            ]
        )

        # Final Linear Projection
        self.fc = nn.Linear(self.fusion_size, Config.num_targets)

        # Initialize the new layers
        self._init_weights(self.fc)
        self._init_weights(self.layer_norm)

    def _init_weights(self, module):
        """Initialize weights for the head layers."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def pool(self, hidden_state, mask):
        """
        Performs Masked Global Average and Max Pooling.

        Args:
            hidden_state: [batch_size, seq_len, hidden_size]
            mask: [batch_size, seq_len]

        Returns:
            avg_pool: [batch_size, hidden_size]
            max_pool: [batch_size, hidden_size]
        """
        # Expand mask to match hidden_state dimensions: [batch, seq, hidden]
        mask_expanded = mask.unsqueeze(-1).expand(hidden_state.size()).float()

        # --- Average Pooling ---
        # Sum of hidden states where mask is active
        sum_embeddings = torch.sum(hidden_state * mask_expanded, dim=1)
        # Count of active tokens
        sum_mask = mask_expanded.sum(dim=1)
        # Avoid division by zero
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        avg_pool = sum_embeddings / sum_mask

        # --- Max Pooling ---
        # Set padded positions to a very small number so they aren't selected by max
        # Clone to avoid in-place modification errors during backprop
        hidden_state_masked = hidden_state.clone()
        hidden_state_masked[mask_expanded == 0] = -1e9
        max_pool = torch.max(hidden_state_masked, dim=1)[0]

        return avg_pool, max_pool

    def forward(self, q_input_ids, q_attention_mask, a_input_ids, a_attention_mask):
        """
        Forward pass for the dual-encoder.

        Args:
            q_input_ids, q_attention_mask: Question inputs
            a_input_ids, a_attention_mask: Answer inputs
        """
        # 1. Process Question Branch
        q_out = self.backbone(input_ids=q_input_ids, attention_mask=q_attention_mask)
        q_hidden = q_out.last_hidden_state
        q_avg, q_max = self.pool(q_hidden, q_attention_mask)

        # 2. Process Answer Branch
        a_out = self.backbone(input_ids=a_input_ids, attention_mask=a_attention_mask)
        a_hidden = a_out.last_hidden_state
        a_avg, a_max = self.pool(a_hidden, a_attention_mask)

        # 3. Compute Interactions (on Average Pooled vectors)
        inter_prod = q_avg * a_avg
        inter_diff = torch.abs(q_avg - a_avg)
        inter_cos = torch.nn.functional.cosine_similarity(
            q_avg, a_avg, dim=1
        ).unsqueeze(1)

        # 4. Feature Fusion
        # Concatenate: Avg Pooled, Max Pooled, Interactions
        fused_vector = torch.cat(
            [q_avg, a_avg, q_max, a_max, inter_prod, inter_diff, inter_cos], dim=1
        )

        # 5. Normalization
        fused_vector = self.layer_norm(fused_vector)

        # 6. Multi-Sample Dropout Head
        # Pass through K dropout masks and average the logits
        logits_list = []
        for dropout in self.dropouts:
            x = dropout(fused_vector)
            logits = self.fc(x)
            logits_list.append(logits)

        # Stack and average
        final_logits = torch.mean(torch.stack(logits_list), dim=0)

        return final_logits
