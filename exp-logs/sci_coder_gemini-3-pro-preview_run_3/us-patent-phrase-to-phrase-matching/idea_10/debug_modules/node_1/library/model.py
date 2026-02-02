import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from library.config import CFG


class DynamicLayerMixing(nn.Module):
    """
    Computes a weighted sum of all hidden layers using learnable scalar weights.
    Formula: H_mix = sum(exp(w_i) / sum(exp(w_j)) * H_i)
    """

    def __init__(self, n_layers):
        super().__init__()
        self.n_layers = n_layers
        # Initialize weights to 0 (resulting in uniform attention after softmax initially)
        self.layer_weights = nn.Parameter(torch.zeros(n_layers))

    def forward(self, hidden_states):
        # hidden_states: Tuple of (batch_size, seq_len, hidden_dim)
        # Stack to: (n_layers, batch_size, seq_len, hidden_dim)
        all_layers = torch.stack(hidden_states)

        # Calculate softmax weights
        # weights: (n_layers) -> (n_layers, 1, 1, 1) for broadcasting
        weights = torch.softmax(self.layer_weights, dim=0).view(-1, 1, 1, 1)

        # Weighted sum
        mixed_layer = torch.sum(all_layers * weights, dim=0)
        return mixed_layer


class AttentionPooling(nn.Module):
    """
    Aggregates the sequence of hidden states into a single vector using attention.
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, last_hidden_state, attention_mask):
        # last_hidden_state: (batch_size, seq_len, hidden_dim)
        # attention_mask: (batch_size, seq_len)

        # Compute raw attention scores
        w = self.attention(last_hidden_state)  # (batch_size, seq_len, 1)

        # Mask padding tokens to exclude them from softmax
        if attention_mask is not None:
            # Create a large negative value for padding tokens
            # attention_mask is 1 for valid, 0 for pad
            padding_mask = (1.0 - attention_mask.unsqueeze(-1)) * -1e9
            w = w + padding_mask

        # Normalize weights
        weights = torch.softmax(w, dim=1)

        # Weighted sum of hidden states
        feature = torch.sum(
            weights * last_hidden_state, dim=1
        )  # (batch_size, hidden_dim)
        return feature


class CustomModel(nn.Module):
    """
    Main model architecture combining DeBERTa backbone, Dynamic Layer Mixing,
    Attention Pooling, and Dual Heads (Regression + Classification).
    """

    def __init__(self, cfg=CFG, pretrained=True):
        super().__init__()
        self.cfg = cfg
        self.config = AutoConfig.from_pretrained(
            cfg.model_name, output_hidden_states=True
        )

        # Load Backbone
        if pretrained:
            self.model = AutoModel.from_pretrained(cfg.model_name, config=self.config)
        else:
            self.model = AutoModel.from_config(self.config)

        # Determine number of layers (hidden layers + embedding layer)
        # DeBERTa output_hidden_states returns N+1 tensors
        self.n_hidden_layers = self.config.num_hidden_layers + 1

        # Components
        self.layer_mixing = DynamicLayerMixing(self.n_hidden_layers)
        self.pooler = AttentionPooling(self.config.hidden_size)

        # Multi-Sample Dropout
        # Using 5 dropout layers as is common in Kaggle NLP competitions
        self.dropouts = nn.ModuleList([nn.Dropout(0.1) for _ in range(5)])

        # Heads
        # Regression Head: outputs a single similarity score
        self.fc_reg = nn.Linear(self.config.hidden_size, 1)

        # Classification Head: outputs logits for 5 classes (0, 0.25, 0.5, 0.75, 1.0)
        self.fc_class = nn.Linear(self.config.hidden_size, 5)

        # Initialize custom layers
        self._init_weights(self.pooler)
        self._init_weights(self.fc_reg)
        self._init_weights(self.fc_class)

    def _init_weights(self, module):
        """Initialize weights for new layers similar to transformer defaults."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        # 1. Backbone Forward Pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # 2. Dynamic Layer Mixing
        # outputs.hidden_states is a tuple of (batch, seq, dim) tensors
        mixed_features = self.layer_mixing(outputs.hidden_states)

        # 3. Attention Pooling
        pooled_features = self.pooler(mixed_features, attention_mask)

        # 4. Multi-Sample Dropout & Heads
        reg_outputs = []
        class_outputs = []

        for dropout in self.dropouts:
            dropped = dropout(pooled_features)

            # Regression prediction
            reg_outputs.append(self.fc_reg(dropped))

            # Classification logits
            class_outputs.append(self.fc_class(dropped))

        # Average the predictions across dropout samples
        final_score = torch.mean(torch.stack(reg_outputs), dim=0)  # Shape: (batch, 1)
        final_logits = torch.mean(
            torch.stack(class_outputs), dim=0
        )  # Shape: (batch, 5)

        return {"score": final_score, "logits": final_logits}
