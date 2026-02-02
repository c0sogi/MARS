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

    def _pool(self, last_hidden_state, mask, pool_type="avg"):
        """
        Args:
            last_hidden_state: (Batch, Seq, Hidden)
            mask: (Batch, Seq) - Binary mask (1 for valid, 0 for ignore)
            pool_type: 'avg' or 'max'
        """
        # Expand mask to (Batch, Seq, 1) for broadcasting
        mask_expanded = mask.unsqueeze(-1).float()

        if pool_type == "avg":
            # Sum of embeddings
            sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
            # Count of tokens (clamp to avoid div by zero)
            sum_mask = torch.sum(mask_expanded, dim=1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            return sum_embeddings / sum_mask

        elif pool_type == "max":
            # Fill masked positions with very small number
            # mask == 0 means ignore
            input_masked = last_hidden_state.masked_fill(mask_expanded == 0, -1e9)
            max_embeddings = torch.max(input_masked, dim=1)[0]
            return max_embeddings

        else:
            raise ValueError("Invalid pool_type")

    def forward(self, input_ids, attention_mask, question_mask, answer_mask, **kwargs):
        # Cite debug_lesson_5: Synchronize Tokenization Strategy with Model Input Signature

        # Single Encoder Forward Pass
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state

        # Pooling (Hybrid Avg + Max) using Partition Masks
        # Cite solution_lesson_node_00003: Masked Dual-Pooling
        q_avg = self._pool(last_hidden_state, question_mask, "avg")
        q_max = self._pool(last_hidden_state, question_mask, "max")

        a_avg = self._pool(last_hidden_state, answer_mask, "avg")
        a_max = self._pool(last_hidden_state, answer_mask, "max")

        # Interaction Terms (based on Avg pooling)
        interaction_prod = q_avg * a_avg
        interaction_diff = torch.abs(q_avg - a_avg)

        # Fusion
        fused_vector = torch.cat(
            [q_avg, q_max, a_avg, a_max, interaction_prod, interaction_diff], dim=1
        )

        # Normalize
        fused_vector = self.layer_norm(fused_vector)

        # MLP Head
        # 1. Projection + Activation
        features = self.head_projection(fused_vector)
        features = self.activation(features)

        # 2. Multi-Sample Dropout + Final Regression
        logits_list = []
        for dropout in self.dropouts:
            dropped_features = dropout(features)
            logits_list.append(self.regressor(dropped_features))

        # Average logits across dropout samples
        logits = torch.mean(torch.stack(logits_list), dim=0)

        return logits
