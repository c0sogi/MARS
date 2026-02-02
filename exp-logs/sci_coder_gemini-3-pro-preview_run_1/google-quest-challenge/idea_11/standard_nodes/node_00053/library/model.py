import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config


class MLPHead(nn.Module):
    """
    Standard MLP Head with ReLU activation and Backbone-aligned initialization.
    Cite solution_lesson_node_00051: Initialize custom heads to match backbone (Normal 0.02).
    Cite solution_lesson_node_00027: Necessity of Non-Linear Projection Heads.
    """

    def __init__(self, input_dim, hidden_dim, num_targets, dropout_prob=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(hidden_dim, num_targets)

        self._init_weights()

    def _init_weights(self):
        # Cite solution_lesson_node_00051
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class DistilRoBERTaDualEncoder(nn.Module):
    """
    Task-Adapted DistilRoBERTa Dual-Encoder with MLP Head.

    Features:
    1. Shared DistilRoBERTa backbone.
    2. Masked Hybrid Pooling (Avg + Max).
    3. Interaction-Aware Fusion (Product + Difference on Avg vectors).
    4. MLP Head with Backbone-Aligned Initialization.
    """

    def __init__(self):
        super().__init__()

        # Load Configuration
        self.backbone_name = Config.BACKBONE
        self.hidden_size = Config.HIDDEN_SIZE
        self.num_targets = Config.NUM_TARGETS

        # 1. Backbone
        config = AutoConfig.from_pretrained(self.backbone_name)
        self.backbone = AutoModel.from_pretrained(self.backbone_name, config=config)

        # Dimensions calculation
        # F = [u_avg, u_max, v_avg, v_max, u_avg*v_avg, |u_avg-v_avg|]
        # All components are size hidden_size (768)
        # Total F size = 6 * 768 = 4608
        self.fusion_dim = 6 * self.hidden_size

        # Layer Normalization for the fused vector
        self.fusion_norm = nn.LayerNorm(self.fusion_dim)

        # 2. MLP Head
        self.head = MLPHead(
            input_dim=self.fusion_dim,
            hidden_dim=self.hidden_size,
            num_targets=self.num_targets,
        )

    def _masked_pooling(self, hidden_states, attention_mask):
        """
        Performs Masked Global Average and Max Pooling.
        """
        # hidden_states: [batch, seq_len, hidden]
        # attention_mask: [batch, seq_len]

        # Expand mask to match hidden dimensions
        # mask_expanded: [batch, seq_len, hidden]
        mask_expanded = (
            attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        )

        # --- Average Pooling ---
        # Sum hidden states where mask is 1
        sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
        # Sum mask values (count of tokens)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        avg_pool = sum_embeddings / sum_mask

        # --- Max Pooling ---
        # Set padded positions to large negative value so they aren't selected as max
        # 1e-9 is used for stability in avg, but for max we need effectively -inf
        hidden_states_masked = hidden_states.clone()
        hidden_states_masked[mask_expanded == 0] = -1e9

        max_pool = torch.max(hidden_states_masked, 1)[0]

        return avg_pool, max_pool

    def forward(
        self, q_input_ids, q_attention_mask, a_input_ids, a_attention_mask, **kwargs
    ):
        # --- Question Stream ---
        q_out = self.backbone(input_ids=q_input_ids, attention_mask=q_attention_mask)
        q_hidden = q_out.last_hidden_state
        u_avg, u_max = self._masked_pooling(q_hidden, q_attention_mask)

        # --- Answer Stream ---
        a_out = self.backbone(input_ids=a_input_ids, attention_mask=a_attention_mask)
        a_hidden = a_out.last_hidden_state
        v_avg, v_max = self._masked_pooling(a_hidden, a_attention_mask)

        # --- Interaction-Aware Fusion ---
        # Interactions computed on Average Pooled vectors
        interaction_prod = u_avg * v_avg
        interaction_diff = torch.abs(u_avg - v_avg)

        # Construct Fused Vector F
        # F = [u_avg, u_max, v_avg, v_max, u_avg*v_avg, |u_avg-v_avg|]
        fused_vector = torch.cat(
            [u_avg, u_max, v_avg, v_max, interaction_prod, interaction_diff], dim=1
        )

        # Apply Layer Normalization
        fused_vector = self.fusion_norm(fused_vector)

        # --- Residual Fusion Head ---
        logits = self.head(fused_vector)

        return logits
