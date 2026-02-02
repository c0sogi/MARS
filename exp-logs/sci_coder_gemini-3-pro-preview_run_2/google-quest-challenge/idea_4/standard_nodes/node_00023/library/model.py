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
    - Decoupled Prediction Heads:
        1. Question Head: Predicts 21 question-intrinsic labels from Question embedding (u).
        2. QA Head: Predicts 9 answer-relational labels from Interaction embedding ([u, v, |u-v|, u*v]).
    - Non-Linear MLP Heads for better feature disentanglement.
    """

    def __init__(self):
        super(CausalDebertaSiamese, self).__init__()

        # Load Configuration and Backbone
        self.config = AutoConfig.from_pretrained(Config.MODEL_NAME)
        self.backbone = AutoModel.from_pretrained(Config.MODEL_NAME, config=self.config)

        # Pooling Layer
        self.pooler = MeanPooling()

        self.hidden_size = self.config.hidden_size

        # 1. Question Head (21 targets)
        # Input: Question embedding u (size: hidden_size)
        self.q_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, 21),
        )

        # 2. QA Interaction Head (9 targets)
        # Input: Interaction vector [u, v, |u-v|, u*v] (size: hidden_size * 4)
        interaction_dim = self.hidden_size * 4
        self.qa_head = nn.Sequential(
            nn.Linear(interaction_dim, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, 9),
        )

        # Initialize weights
        self._init_weights(self.q_head)
        self._init_weights(self.qa_head)

    def _init_weights(self, module_list):
        for module in module_list:
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)

    def forward(self, q_input_ids, q_attention_mask, a_input_ids, a_attention_mask):
        """
        Forward pass with causal masking logic.
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

        # 3. Question Head Prediction (First 21 columns)
        q_logits = self.q_head(u)

        # 4. Interaction Head Prediction (Last 9 columns)
        diff = torch.abs(u - v)
        prod = u * v
        interaction_features = torch.cat([u, v, diff, prod], dim=1)
        qa_logits = self.qa_head(interaction_features)

        # 5. Concatenate Logits
        # The target columns are ordered: 21 Question cols, then 9 Answer cols.
        logits = torch.cat([q_logits, qa_logits], dim=1)

        return logits
