import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config


class WeightedLayerPooling(nn.Module):
    """
    Learns a scalar weight for each layer of the transformer to compute a weighted sum.
    Formula: H_mix = sum(softmax(w_i) * H_i)
    """

    def __init__(self, num_hidden_layers, layer_start: int = 0):
        super(WeightedLayerPooling, self).__init__()
        self.layer_start = layer_start
        self.num_hidden_layers = num_hidden_layers
        # Create learnable weights for each layer involved.
        # +1 accounts for the embedding layer which is included in hidden_states
        self.layer_weights = nn.Parameter(
            torch.tensor(
                [1.0] * (num_hidden_layers + 1 - layer_start), dtype=torch.float
            )
        )

    def forward(self, all_hidden_states):
        # all_hidden_states is a tuple of (batch, seq_len, hidden_size) tensors
        # Stack them to shape: (num_layers, batch, seq_len, hidden_size)
        all_layer_embeddings = torch.stack(all_hidden_states)

        # Filter layers if start > 0
        if self.layer_start > 0:
            all_layer_embeddings = all_layer_embeddings[self.layer_start :, :, :, :]

        # Compute softmax weights: (num_layers,)
        weights = torch.softmax(self.layer_weights, dim=0)

        # Reshape for broadcasting: (num_layers, 1, 1, 1)
        weights = weights.view(-1, 1, 1, 1)

        # Compute weighted sum along the layer dimension
        # Output: (batch, seq_len, hidden_size)
        weighted_embeddings = (weights * all_layer_embeddings).sum(dim=0)

        return weighted_embeddings


class AttentionPooling(nn.Module):
    """
    Aggregates a sequence of hidden states into a single vector using attention.
    """

    def __init__(self, hidden_size):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1)
        )

    def forward(self, last_hidden_state, attention_mask):
        # last_hidden_state: (batch, seq_len, hidden_size)
        # attention_mask: (batch, seq_len)

        # Compute raw attention scores: (batch, seq_len, 1)
        w = self.attention(last_hidden_state)

        if attention_mask is not None:
            # Expand mask to match dimensions: (batch, seq_len, 1)
            extended_mask = attention_mask.unsqueeze(-1)
            # Mask padding tokens with a very large negative value so softmax -> 0
            w = w.masked_fill(extended_mask == 0, -1e4)

        # Normalize scores to probabilities
        weights = torch.softmax(w, dim=1)

        # Compute weighted sum: (batch, hidden_size)
        context_vector = torch.sum(weights * last_hidden_state, dim=1)

        return context_vector


class CustomModel(nn.Module):
    """
    DeBERTa-v3-Large with Weighted Layer Pooling, Attention Pooling,
    Multi-Sample Dropout, and Dual Heads (Regression + Classification).
    """

    def __init__(self, cfg: Config):
        super(CustomModel, self).__init__()
        self.cfg = cfg

        # Load Configuration and Backbone
        self.config = AutoConfig.from_pretrained(cfg.model_name)
        self.config.output_hidden_states = True
        self.model = AutoModel.from_pretrained(cfg.model_name, config=self.config)

        # Pooling Layers
        self.pooler = WeightedLayerPooling(
            num_hidden_layers=self.config.num_hidden_layers, layer_start=0
        )
        self.attention_pooler = AttentionPooling(self.config.hidden_size)

        # Multi-Sample Dropout
        self.dropouts = nn.ModuleList(
            [nn.Dropout(cfg.msd_dropout) for _ in range(cfg.num_msd)]
        )

        # Heads
        # Regression Head: Predicts continuous score (0.0 to 1.0)
        self.reg_head = nn.Linear(self.config.hidden_size, 1)

        # Classification Head: Predicts 5 classes (0.0, 0.25, 0.5, 0.75, 1.0)
        self.class_head = nn.Linear(self.config.hidden_size, 5)

        # Initialize weights for custom layers
        self._init_weights(self.attention_pooler)
        self._init_weights(self.reg_head)
        self._init_weights(self.class_head)

    def _init_weights(self, module):
        """Initialize weights for custom layers using standard transformer init."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        # Pass through Backbone
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # Weighted Layer Pooling
        # outputs.hidden_states contains (embeddings + 24 layers)
        sequence_output = self.pooler(outputs.hidden_states)

        # Attention Pooling
        pooled_output = self.attention_pooler(sequence_output, attention_mask)

        # Multi-Sample Dropout & Heads
        reg_logits_list = []
        class_logits_list = []

        for dropout in self.dropouts:
            dropped_output = dropout(pooled_output)

            # Forward through heads
            reg_logits_list.append(self.reg_head(dropped_output))
            class_logits_list.append(self.class_head(dropped_output))

        # Average the predictions from all dropout samples
        reg_logits = torch.mean(torch.stack(reg_logits_list), dim=0)
        class_logits = torch.mean(torch.stack(class_logits_list), dim=0)

        return {"logits": reg_logits, "class_logits": class_logits}
