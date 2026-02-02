import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config


class MeanPooling(nn.Module):
    """
    Performs mean pooling on the token embeddings, accounting for the attention mask.
    """

    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        # Expand attention mask to match the embedding dimensions
        # Mask shape: [batch_size, seq_len] -> [batch_size, seq_len, hidden_size]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )

        # Sum embeddings over the sequence length where mask is 1
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)

        # Sum the mask values to get the count of valid tokens
        sum_mask = input_mask_expanded.sum(1)

        # Avoid division by zero by clamping the divisor
        sum_mask = torch.clamp(sum_mask, min=1e-9)

        # Compute mean
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class CausalDebertaSiamese(nn.Module):
    """
    Causal-Aware DeBERTa-v3 Siamese Network.

    Features:
    - Shared DeBERTa-v3 backbone.
    - Decoupled Prediction Heads (Cite solution_lesson_node_00014):
        1. Question Head: Predicts 21 Q-targets using only Question embedding (u).
        2. Answer Head: Predicts 9 A-targets using Interaction features [u, v, |u-v|, u*v].
    """

    def __init__(self):
        super(CausalDebertaSiamese, self).__init__()

        # Load Configuration and Backbone
        self.config = AutoConfig.from_pretrained(Config.MODEL_NAME)
        self.backbone = AutoModel.from_pretrained(Config.MODEL_NAME, config=self.config)

        # Pooling Layer
        self.pooler = MeanPooling()

        self.hidden_size = self.config.hidden_size
        interaction_dim = self.hidden_size * 4

        # 1. Question-Intrinsic Head (21 labels)
        # Input: u (Question Embedding)
        self.q_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, Config.NUM_Q_LABELS),
        )

        # 2. Relational Answer Head (9 labels)
        # Input: [u, v, |u-v|, u*v]
        self.a_head = nn.Sequential(
            nn.Linear(interaction_dim, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, Config.NUM_A_LABELS),
        )

        # Initialize weights
        self._init_weights(self.q_head)
        self._init_weights(self.a_head)

    def _init_weights(self, module):
        """
        Recursively initialize weights.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Sequential):
            for m in module:
                self._init_weights(m)

    def forward(self, q_input_ids, q_attention_mask, a_input_ids, a_attention_mask):
        """
        Forward pass with decoupled heads.
        """
        # 1. Process Question Stream
        q_outputs = self.backbone(
            input_ids=q_input_ids, attention_mask=q_attention_mask
        )
        u = self.pooler(q_outputs.last_hidden_state, q_attention_mask)

        # 2. Process Answer Stream
        a_outputs = self.backbone(
            input_ids=a_input_ids, attention_mask=a_attention_mask
        )
        v = self.pooler(a_outputs.last_hidden_state, a_attention_mask)

        # 3. Question Head Prediction (Causal Constraint)
        q_logits = self.q_head(u)

        # 4. Interaction Features
        diff = torch.abs(u - v)
        prod = u * v
        features = torch.cat([u, v, diff, prod], dim=1)

        # 5. Answer Head Prediction
        a_logits = self.a_head(features)

        # 6. Concatenate Logits [Q_logits (21), A_logits (9)] -> Total 30
        return torch.cat([q_logits, a_logits], dim=1)
