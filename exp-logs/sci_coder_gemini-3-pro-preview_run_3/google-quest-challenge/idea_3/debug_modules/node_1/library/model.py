import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import ModelConfig


class SegmentAwareCrossEncoder(nn.Module):
    """
    A Cross-Encoder architecture that extracts segment-specific features
    (Question vs Answer) from the joint attention context.

    Features extracted:
    1. h_cls: Global context ([CLS] token)
    2. h_q: Mean pooled representation of Question tokens
    3. h_a: Mean pooled representation of Answer tokens
    4. h_diff: Absolute difference |h_q - h_a|

    Total Feature Dimension: 4 * hidden_size
    """

    def __init__(self):
        super().__init__()

        # Load Configuration
        self.config = AutoConfig.from_pretrained(ModelConfig.model_name)
        self.config.hidden_dropout_prob = ModelConfig.dropout
        self.config.attention_probs_dropout_prob = ModelConfig.dropout

        # Load Backbone
        self.backbone = AutoModel.from_pretrained(
            ModelConfig.model_name, config=self.config
        )

        # Feature Dimensions
        self.hidden_size = ModelConfig.hidden_size
        self.feature_dim = self.hidden_size * 4  # [CLS, Q, A, Diff]

        # Classification Head for Stage 1 (Fine-tuning)
        self.head = nn.Linear(self.feature_dim, ModelConfig.num_labels)

        # Initialize weights of the head
        self._init_weights(self.head)

    def _init_weights(self, module):
        """Initialize weights for the linear head."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def _masked_mean_pooling(self, hidden_state, mask):
        """
        Computes mean pooling of hidden states based on a specific mask.

        Args:
            hidden_state: [batch_size, seq_len, hidden_size]
            mask: [batch_size, seq_len] (1 for relevant tokens, 0 otherwise)

        Returns:
            pooled: [batch_size, hidden_size]
        """
        # Expand mask to match hidden_state dimensions
        # mask: [batch, seq_len] -> [batch, seq_len, 1]
        mask_expanded = mask.unsqueeze(-1).expand(hidden_state.size()).float()

        # Sum hidden states corresponding to the mask
        sum_embeddings = torch.sum(hidden_state * mask_expanded, dim=1)

        # Sum mask values to get token counts (clamp to avoid division by zero)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)

        # Mean pooling
        return sum_embeddings / sum_mask

    def forward(self, input_ids, attention_mask, q_mask, a_mask, labels=None):
        """
        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            q_mask: [batch, seq_len] - 1 for Question tokens, 0 otherwise
            a_mask: [batch, seq_len] - 1 for Answer tokens, 0 otherwise
            labels: Optional, not used inside model but kept for API consistency

        Returns:
            logits: [batch, num_labels]
            features: [batch, feature_dim]
        """
        # Pass through backbone
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # [batch, seq_len, hidden]

        # 1. Extract [CLS] embedding (index 0)
        h_cls = last_hidden_state[:, 0, :]

        # 2. Extract Contextualized Question Embedding (Mean Pool over q_mask)
        h_q = self._masked_mean_pooling(last_hidden_state, q_mask)

        # 3. Extract Contextualized Answer Embedding (Mean Pool over a_mask)
        h_a = self._masked_mean_pooling(last_hidden_state, a_mask)

        # 4. Interaction Term
        h_diff = torch.abs(h_q - h_a)

        # Concatenate all features
        features = torch.cat([h_cls, h_q, h_a, h_diff], dim=1)

        # Compute Logits
        logits = self.head(features)

        return logits, features
