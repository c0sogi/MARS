import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config


class LinearAttentionPooling(nn.Module):
    """
    Implementation of Linear Attention Pooling.
    Computes a weighted sum of hidden states based on learned attention scores.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, last_hidden_state, attention_mask):
        # last_hidden_state: (batch_size, seq_len, hidden_size)
        # attention_mask: (batch_size, seq_len)

        # 1. Compute raw attention scores
        # Shape: (batch_size, seq_len, 1)
        weights = self.attention(last_hidden_state)

        # 2. Mask padding tokens
        # We set the score of padding tokens to a very large negative number
        # so that the softmax probability is effectively 0.
        mask_expanded = attention_mask.unsqueeze(-1)  # (batch_size, seq_len, 1)
        weights = weights.masked_fill(mask_expanded == 0, -1e9)

        # 3. Normalize scores to probabilities
        att_weights = torch.softmax(weights, dim=1)

        # 4. Compute weighted sum
        # (batch, seq, 1) * (batch, seq, hidden) -> (batch, seq, hidden)
        # Sum over sequence dimension -> (batch, hidden)
        context_vector = torch.sum(att_weights * last_hidden_state, dim=1)
        return context_vector


class ToxicityModel(nn.Module):
    """
    Context-Aware DeBERTa-v3 with Multi-Sample Dropout.
    """

    def __init__(self):
        super().__init__()
        self.config = Config()

        # Load Configuration and Backbone
        # We use AutoConfig/AutoModel to load DeBERTa-v3-base
        model_config = AutoConfig.from_pretrained(self.config.model_name)
        self.backbone = AutoModel.from_pretrained(
            self.config.model_name, config=model_config
        )

        hidden_size = model_config.hidden_size

        # Pooling Mechanism 1: Linear Attention
        self.linear_attention = LinearAttentionPooling(hidden_size)

        # Multi-Sample Dropout
        # Create multiple dropout layers with the same rate
        self.dropouts = nn.ModuleList(
            [
                nn.Dropout(self.config.dropout_rate)
                for _ in range(self.config.n_dropout_samples)
            ]
        )

        # Classification Head
        # Input dimension is 2 * hidden_size because we concatenate:
        # 1. Linear Attention Pooling Output
        # 2. Global Max Pooling Output
        self.fc = nn.Linear(hidden_size * 2, self.config.num_classes)

        # Initialize weights for the new layers
        self._init_weights(self.linear_attention.attention)
        self._init_weights(self.fc)

    def _init_weights(self, module):
        """
        Standard weight initialization for linear layers.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask):
        # 1. Backbone Forward Pass
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # (batch, seq_len, hidden_size)

        # 2. Hybrid Pooling

        # A. Linear Attention Pooling
        att_vector = self.linear_attention(last_hidden_state, attention_mask)

        # B. Global Max Pooling
        # Mask padding tokens with -inf before taking max
        mask_expanded = attention_mask.unsqueeze(-1)
        masked_hidden = last_hidden_state.masked_fill(mask_expanded == 0, -1e9)
        max_vector = torch.max(masked_hidden, dim=1)[0]

        # Concatenate features
        features = torch.cat([att_vector, max_vector], dim=1)

        # 3. Multi-Sample Dropout & Classification
        logits_list = []
        for dropout in self.dropouts:
            # Apply dropout
            dropped_features = dropout(features)
            # Apply classifier (shared weights)
            logits = self.fc(dropped_features)
            logits_list.append(logits)

        # 4. Average Logits
        # Stack logits -> (n_samples, batch, num_classes)
        # Mean over dim 0 -> (batch, num_classes)
        final_logits = torch.stack(logits_list).mean(dim=0)

        return final_logits
