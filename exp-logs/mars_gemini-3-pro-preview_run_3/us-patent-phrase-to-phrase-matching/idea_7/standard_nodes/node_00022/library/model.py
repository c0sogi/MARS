import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from library.config import Config


class AttentionPool(nn.Module):
    """
    Attention Pooling Layer.
    Computes a weighted average of token embeddings based on a learned attention mechanism.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1)
        )

    def forward(self, x, mask):
        """
        Args:
            x: Hidden states (batch_size, seq_len, hidden_size)
            mask: Attention mask (batch_size, seq_len)
        Returns:
            Pooled representation (batch_size, hidden_size)
        """
        # Compute attention scores: (batch, seq, 1)
        w = self.attention(x)

        # Mask padding tokens so they don't contribute to the average
        # mask is 1 for tokens, 0 for padding.
        # We want to set padding positions to -infinity before softmax.
        extended_mask = (1.0 - mask.unsqueeze(-1)) * -1e9
        w = w + extended_mask

        # Compute weights
        w = torch.softmax(w, dim=1)

        # Weighted sum
        # (batch, seq, 1) * (batch, seq, hidden) -> (batch, seq, hidden) -> sum -> (batch, hidden)
        return torch.sum(x * w, dim=1)


class CustomDeberta(nn.Module):
    """
    Hierarchical Context-Aware Cross-Encoder with Multi-Layer Pooling and MSD.
    """

    def __init__(self, model_path=None, pretrained=True):
        super().__init__()

        # Determine model path (default to Config or override)
        self.model_path = model_path if model_path else Config.model_backbone

        # Load Configuration
        self.config = AutoConfig.from_pretrained(self.model_path)
        self.config.output_hidden_states = True  # Required for Multi-Layer Pooling

        # Load Backbone
        if pretrained:
            self.backbone = AutoModel.from_pretrained(
                self.model_path, config=self.config
            )
        else:
            self.backbone = AutoModel.from_config(self.config)

        # Dimensions
        self.hidden_size = self.config.hidden_size
        self.pool_layers = Config.pool_layers

        # Multi-Layer Attention Pooling
        # We create a separate AttentionPool for each layer we want to aggregate
        self.poolers = nn.ModuleList(
            [AttentionPool(self.hidden_size) for _ in range(self.pool_layers)]
        )

        # The concatenated output size
        self.concat_hidden_size = self.hidden_size * self.pool_layers

        # Multi-Sample Dropout (MSD)
        self.dropouts = nn.ModuleList(
            [
                nn.Dropout(Config.multi_sample_dropout_rate)
                for _ in range(Config.multi_sample_dropout_num)
            ]
        )

        # Output Heads
        # Regression Head (Continuous Score)
        self.fc_reg = nn.Linear(self.concat_hidden_size, Config.target_size)

        # Classification Head (Discrete Bins)
        self.fc_cls = nn.Linear(self.concat_hidden_size, Config.num_classes)

        # Initialize weights for new layers
        self._init_weights(self.poolers)
        self._init_weights(self.fc_reg)
        self._init_weights(self.fc_cls)

    def _init_weights(self, module):
        """
        Initialize weights for specific modules using Xavier Uniform.
        """
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.ModuleList):
            for m in module:
                self._init_weights(m)
        elif isinstance(module, nn.Sequential):
            for m in module:
                self._init_weights(m)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """
        Forward pass.

        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            token_type_ids: (batch, seq_len) - Optional, depending on tokenizer

        Returns:
            reg_logits: (batch, 1)
            cls_logits: (batch, num_classes)
        """
        # Pass through backbone
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # Get all hidden states (tuple of tensors)
        all_hidden_states = outputs.hidden_states

        # Extract the last N layers
        # all_hidden_states is usually (embedding, layer1, ... layerN)
        # We want the last 'pool_layers'
        selected_layers = all_hidden_states[-self.pool_layers :]

        pooled_outputs = []

        # Apply Attention Pooling to each selected layer
        for i, layer_hidden_state in enumerate(selected_layers):
            # layer_hidden_state: (batch, seq, hidden)
            pooled_layer = self.poolers[i](layer_hidden_state, attention_mask)
            pooled_outputs.append(pooled_layer)

        # Concatenate pooled representations
        # Shape: (batch, hidden * pool_layers)
        concat_output = torch.cat(pooled_outputs, dim=1)

        # Multi-Sample Dropout & Heads
        reg_logits_sum = 0
        cls_logits_sum = 0

        for dropout in self.dropouts:
            dropped_output = dropout(concat_output)

            reg_logits_sum += self.fc_reg(dropped_output)
            cls_logits_sum += self.fc_cls(dropped_output)

        # Average the predictions
        reg_logits = reg_logits_sum / len(self.dropouts)
        cls_logits = cls_logits_sum / len(self.dropouts)

        return reg_logits, cls_logits
