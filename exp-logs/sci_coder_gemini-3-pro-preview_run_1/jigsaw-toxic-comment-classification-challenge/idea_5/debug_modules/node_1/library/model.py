import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config


class WeightedLayerPooling(nn.Module):
    """
    Computes a learnable weighted average of the last `num_hidden_layers` hidden states.
    This allows the model to dynamically select the most relevant layers for the task.
    """

    def __init__(self, num_hidden_layers):
        super(WeightedLayerPooling, self).__init__()
        self.num_hidden_layers = num_hidden_layers
        # Initialize weights to be equal (softmax will make them 1/N)
        self.layer_weights = nn.Parameter(torch.ones(num_hidden_layers))

    def forward(self, all_hidden_states):
        # all_hidden_states is a tuple of tensors from the backbone
        # We select the last 'num_hidden_layers'
        selected_layers = all_hidden_states[-self.num_hidden_layers :]

        # Compute softmax weights
        weights = torch.softmax(self.layer_weights, dim=0)

        # Compute weighted sum
        # We iterate to avoid stacking large tensors if memory is constrained,
        # though on A100 stacking is likely fine. Iteration is safe.
        weighted_output = selected_layers[0] * weights[0]
        for i in range(1, self.num_hidden_layers):
            weighted_output += selected_layers[i] * weights[i]

        return weighted_output


class LinearAttentionPooling(nn.Module):
    """
    Applies Linear Attention Pooling to the sequence to extract a context-aware vector.
    """

    def __init__(self, hidden_size):
        super(LinearAttentionPooling, self).__init__()
        self.attention_linear = nn.Linear(hidden_size, 1)

    def forward(self, x, attention_mask):
        # x: (batch, seq_len, hidden_size)
        # attention_mask: (batch, seq_len)

        # Calculate attention scores
        # w: (batch, seq_len, 1)
        w = self.attention_linear(x)

        if attention_mask is not None:
            # Mask padding tokens with a large negative value
            # attention_mask is 1 for keep, 0 for pad
            mask_value = -1e9
            extended_mask = (1.0 - attention_mask.unsqueeze(-1)) * mask_value
            w = w + extended_mask

        # Softmax to get normalized weights
        att_weights = torch.softmax(w, dim=1)

        # Weighted sum of the sequence
        # (batch, seq_len, 1) * (batch, seq_len, hidden_size) -> sum over seq dim
        weighted_output = torch.sum(att_weights * x, dim=1)

        return weighted_output


class ToxicityModel(nn.Module):
    """
    Main model class for Toxicity Classification.
    Architecture:
    1. Backbone: DeBERTa-v3-large
    2. Pooler: Weighted Layer Aggregation (Last 4 layers)
    3. Head: Hybrid Pooling (Linear Attention + Max) -> Multi-Sample Dropout -> Linear
    """

    def __init__(self):
        super(ToxicityModel, self).__init__()

        # Load Configuration
        self.config = AutoConfig.from_pretrained(Config.model_name)
        self.config.output_hidden_states = True

        # Load Backbone
        self.model = AutoModel.from_pretrained(Config.model_name, config=self.config)

        # Initialize Custom Layers
        self.layer_pooler = WeightedLayerPooling(Config.num_layers_to_aggregate)
        self.att_pooler = LinearAttentionPooling(Config.hidden_size)

        # Calculate output dimension: Attention Pool + Max Pool
        self.output_dim = Config.hidden_size * 2

        # Multi-Sample Dropout
        self.use_msd = Config.use_msd
        if self.use_msd:
            self.dropouts = nn.ModuleList(
                [nn.Dropout(Config.msd_dropout) for _ in range(Config.num_msd_heads)]
            )
        else:
            self.dropout = nn.Dropout(Config.fc_dropout)

        # Final Classification Layer
        self.fc = nn.Linear(self.output_dim, Config.num_classes)

        # Initialize weights for custom layers
        self._init_weights(self.att_pooler)
        self._init_weights(self.fc)

    def _init_weights(self, module):
        """
        Initialize weights using the backbone's initializer range.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask):
        # Pass through backbone
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        all_hidden_states = outputs.hidden_states

        # 1. Weighted Layer Aggregation
        # Returns: (batch, seq_len, hidden_size)
        sequence_output = self.layer_pooler(all_hidden_states)

        # 2. Hybrid Pooling
        # A. Linear Attention Pooling
        att_pool = self.att_pooler(sequence_output, attention_mask)

        # B. Global Max Pooling
        # We must mask padding tokens to -inf so they don't affect the max
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(sequence_output.size()).float()
        )
        sequence_output_masked = sequence_output.clone()
        sequence_output_masked[input_mask_expanded == 0] = -1e9
        max_pool = torch.max(sequence_output_masked, 1)[0]

        # Concatenate pooling results
        features = torch.cat([att_pool, max_pool], dim=1)

        # 3. Multi-Sample Dropout & Classification
        if self.use_msd:
            logits_list = []
            for dropout in self.dropouts:
                logits_list.append(self.fc(dropout(features)))
            # Average the predictions across all dropout heads
            logits = torch.mean(torch.stack(logits_list), dim=0)
        else:
            features = self.dropout(features)
            logits = self.fc(features)

        return logits
