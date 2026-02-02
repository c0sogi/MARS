import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config


class QuestModel(nn.Module):
    def __init__(self):
        super(QuestModel, self).__init__()

        # Load Backbone
        self.config = AutoConfig.from_pretrained(Config.model_name)
        self.backbone = AutoModel.from_pretrained(Config.model_name)

        # Enable gradient checkpointing if needed for memory efficiency (optional, but good practice)
        # self.backbone.gradient_checkpointing_enable()

        self.hidden_size = Config.hidden_size

        # Feature Engineering Dimensions
        # 1. Q_avg (hidden)
        # 2. Q_max (hidden)
        # 3. A_avg (hidden)
        # 4. A_max (hidden)
        # 5. Q_avg * A_avg (hidden)
        # 6. |Q_avg - A_avg| (hidden)
        # Total = 6 * hidden_size
        self.fusion_dim = 6 * self.hidden_size

        # Layer Norm before head
        self.layer_norm = nn.LayerNorm(self.fusion_dim)

        # Non-Linear MLP Head
        # Structure: Linear -> GELU -> MSD -> Linear
        self.head_projection = nn.Linear(self.fusion_dim, self.hidden_size)
        self.activation = nn.GELU()

        # Multi-Sample Dropout
        self.dropouts = nn.ModuleList(
            [nn.Dropout(p) for p in Config.multi_sample_dropout_rates]
        )

        # Final Regressor
        self.regressor = nn.Linear(self.hidden_size, Config.num_classes)

        # Initialize weights for new layers
        self._init_weights(self.head_projection)
        self._init_weights(self.regressor)
        self.layer_norm.reset_parameters()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _pool(self, last_hidden_state, attention_mask):
        """
        Returns both Avg and Max pooling.
        Args:
            last_hidden_state: (Batch, Seq, Hidden)
            attention_mask: (Batch, Seq)
        """
        mask_expanded = attention_mask.unsqueeze(-1).float()

        # Avg Pooling
        sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        avg_pool = sum_embeddings / sum_mask

        # Max Pooling
        # Mask out padding with large negative value
        input_masked = last_hidden_state.masked_fill(mask_expanded == 0, -1e9)
        max_pool = torch.max(input_masked, dim=1)[0]

        return avg_pool, max_pool

    def forward(
        self, input_ids_q, attention_mask_q, input_ids_a, attention_mask_a, **kwargs
    ):
        # Process Question
        out_q = self.backbone(input_ids=input_ids_q, attention_mask=attention_mask_q)
        q_avg, q_max = self._pool(out_q.last_hidden_state, attention_mask_q)

        # Process Answer
        out_a = self.backbone(input_ids=input_ids_a, attention_mask=attention_mask_a)
        a_avg, a_max = self._pool(out_a.last_hidden_state, attention_mask_a)

        # Interaction Terms (Cite solution_lesson_node_00007: Use Avg for interactions)
        interaction_prod = q_avg * a_avg
        interaction_diff = torch.abs(q_avg - a_avg)

        # Fusion
        fused_vector = torch.cat(
            [q_avg, q_max, a_avg, a_max, interaction_prod, interaction_diff], dim=1
        )

        # Normalize (Cite solution_lesson_node_00009)
        fused_vector = self.layer_norm(fused_vector)

        # MLP Head (Cite solution_lesson_node_00027)
        features = self.head_projection(fused_vector)
        features = self.activation(features)

        # Multi-Sample Dropout + Final Regression
        logits_list = []
        for dropout in self.dropouts:
            dropped_features = dropout(features)
            logits_list.append(self.regressor(dropped_features))

        logits = torch.mean(torch.stack(logits_list), dim=0)

        return logits
