import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config


class AttentionPooling(nn.Module):
    """
    Implementation of Attention Pooling.
    Aggregates a sequence of hidden states into a single vector using a learned attention mechanism.
    """

    def __init__(self, hidden_size):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, last_hidden_state, attention_mask):
        # last_hidden_state: [batch_size, seq_len, hidden_size]
        # attention_mask: [batch_size, seq_len]

        # Calculate attention weights
        w = self.attention(last_hidden_state)  # [batch_size, seq_len, 1]

        # Mask padding tokens (set to large negative value before softmax)
        # attention_mask is 1 for valid tokens, 0 for padding
        # We want to replace 0s with -inf
        w = w.squeeze(-1)  # [batch_size, seq_len]
        w = w.masked_fill(attention_mask == 0, -1e9)

        # Softmax to get probabilities
        weights = torch.softmax(w, dim=-1)  # [batch_size, seq_len]

        # Weighted sum
        # weights: [batch_size, seq_len] -> [batch_size, seq_len, 1]
        weights = weights.unsqueeze(-1)
        feature_vector = torch.sum(
            last_hidden_state * weights, dim=1
        )  # [batch_size, hidden_size]

        return feature_vector


class ScalarMixingModel(nn.Module):
    """
    DeBERTa-v3 model with Scalar Layer Mixing, Attention Pooling, and Dual Heads.
    """

    def __init__(self, pretrained=True):
        super(ScalarMixingModel, self).__init__()

        # 1. Load Configuration and Backbone
        self.config = AutoConfig.from_pretrained(Config.model_name)
        self.config.output_hidden_states = True  # Crucial for Scalar Mixing
        self.config.hidden_dropout_prob = Config.hidden_dropout_prob
        self.config.attention_probs_dropout_prob = Config.attention_probs_dropout_prob

        if pretrained:
            self.backbone = AutoModel.from_pretrained(
                Config.model_name, config=self.config
            )
        else:
            self.backbone = AutoModel.from_config(self.config)

        # 2. Scalar Layer Mixing
        # DeBERTa Large usually has 24 layers + 1 embedding layer = 25 hidden states
        self.num_hidden_layers = self.config.num_hidden_layers
        # Weights for mixing layers (initialized to 0 -> uniform distribution after softmax)
        self.layer_weights = nn.Parameter(torch.zeros(self.num_hidden_layers + 1))

        # 3. Pooling
        self.pooler = AttentionPooling(self.config.hidden_size)

        # 4. Multi-Sample Dropout
        self.dropout_ops = nn.ModuleList(
            [nn.Dropout(Config.hidden_dropout_prob) for _ in range(5)]
        )

        # 5. Dual Heads
        # Regression Head (Score)
        self.fc = nn.Linear(self.config.hidden_size, 1)

        # Classification Head (Auxiliary)
        self.classifier = nn.Linear(self.config.hidden_size, Config.num_aux_classes)

        # Initialize weights for new layers
        self._init_weights(self.fc)
        self._init_weights(self.classifier)
        # Note: AttentionPooling layers are initialized by PyTorch defaults, which is usually fine,
        # but we can explicitly init if needed.

    def _init_weights(self, module):
        """
        Initialize weights for linear layers similar to Transformers.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        # 1. Backbone Forward
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # outputs.hidden_states is a tuple of (num_layers + 1) tensors
        # Each tensor: [batch_size, seq_len, hidden_size]
        all_hidden_states = outputs.hidden_states

        # 2. Scalar Layer Mixing
        # Stack hidden states: [num_layers+1, batch_size, seq_len, hidden_size]
        stacked_layers = torch.stack(all_hidden_states)

        # Compute layer weights
        # softmax(layer_weights): [num_layers+1]
        # Reshape for broadcasting: [num_layers+1, 1, 1, 1]
        weights = torch.softmax(self.layer_weights, dim=0).view(-1, 1, 1, 1)

        # Weighted sum
        # mixed_layer: [batch_size, seq_len, hidden_size]
        mixed_layer = torch.sum(stacked_layers * weights, dim=0)

        # 3. Pooling
        feature_vector = self.pooler(mixed_layer, attention_mask)

        # 4. Multi-Sample Dropout & Heads
        # We average the outputs of the heads across different dropout masks
        logits_sum = 0
        aux_logits_sum = 0

        for dropout_op in self.dropout_ops:
            dropped_features = dropout_op(feature_vector)
            logits_sum += self.fc(dropped_features)
            aux_logits_sum += self.classifier(dropped_features)

        logits = logits_sum / len(self.dropout_ops)
        aux_logits = aux_logits_sum / len(self.dropout_ops)

        # Flatten logits for regression output [batch_size]
        return logits.squeeze(-1), aux_logits
