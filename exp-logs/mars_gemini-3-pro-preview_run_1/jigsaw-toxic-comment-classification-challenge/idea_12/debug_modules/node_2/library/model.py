import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel
from library.config import Config


class WeightedLayerPooling(nn.Module):
    """
    Computes a learnable weighted sum of the last N encoder layers.
    """

    def __init__(self, num_hidden_layers=12, layer_start=9):
        super(WeightedLayerPooling, self).__init__()
        self.layer_start = layer_start
        self.num_hidden_layers = num_hidden_layers
        # Initialize weights for the selected layers (e.g., last 4)
        self.weights = nn.Parameter(torch.zeros(num_hidden_layers + 1 - layer_start))

    def forward(self, all_hidden_states):
        # all_hidden_states is a tuple of tensors (embedding + layers)
        # Stack the desired layers: [num_selected, batch, seq, hidden]
        selected_layers = torch.stack(all_hidden_states[self.layer_start :])

        # Apply softmax to normalize weights
        w = F.softmax(self.weights, dim=0)

        # Reshape for broadcasting: [num_selected, 1, 1, 1]
        w = w.view(-1, 1, 1, 1)

        # Compute weighted sum
        weighted_sum = (w * selected_layers).sum(dim=0)
        return weighted_sum


class LinearAttentionPooling(nn.Module):
    """
    Computes a context-aware weighted average using Linear Attention (w^T * h).
    """

    def __init__(self, hidden_dim):
        super(LinearAttentionPooling, self).__init__()
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, last_hidden_state, attention_mask):
        # last_hidden_state: [batch, seq, hidden]
        # attention_mask: [batch, seq]

        # Project hidden states to scores: [batch, seq, 1]
        logits = self.linear(last_hidden_state)

        # Mask padding tokens (set to -inf so softmax becomes 0)
        mask = attention_mask.unsqueeze(-1)
        logits = logits.masked_fill(mask == 0, -1e9)

        # Compute attention weights
        weights = torch.softmax(logits, dim=1)

        # Weighted sum: [batch, hidden]
        weighted_avg = torch.sum(weights * last_hidden_state, dim=1)
        return weighted_avg


class CustomModel(nn.Module):
    """
    DeBERTa-v3 based model with Weighted Layer Pooling, Hybrid Head, and Multi-Sample Dropout.
    """

    def __init__(self, pretrained=True):
        super().__init__()
        self.config = AutoConfig.from_pretrained(
            Config.model_name, output_hidden_states=True
        )

        # Load backbone
        if pretrained:
            self.model = AutoModel.from_pretrained(
                Config.model_name, config=self.config
            )
        else:
            self.model = AutoModel.from_config(self.config)

        if Config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        # Weighted Layer Pooling (Last 4 layers)
        # DeBERTa base has 12 layers (indices 0-11) + embedding. output_hidden_states has 13 tensors.
        # We want indices 9, 10, 11, 12.
        # layer_start = 12 - 3 = 9.
        self.layer_pooler = WeightedLayerPooling(
            num_hidden_layers=self.config.num_hidden_layers,
            layer_start=self.config.num_hidden_layers - 3,
        )

        # Pooling Heads
        self.attention_pooler = LinearAttentionPooling(self.config.hidden_size)

        # Classification Head
        # Concatenating Attention Pooling + Global Max Pooling -> 2 * hidden_size
        self.fc = nn.Linear(self.config.hidden_size * 2, Config.num_labels)

        # Multi-Sample Dropout (MSD)
        self.dropouts = nn.ModuleList(
            [nn.Dropout(Config.msd_dropout) for _ in range(Config.msd_num)]
        )

        self._init_weights(self.fc)
        self._init_weights(self.attention_pooler.linear)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask, labels=None):
        # Backbone Forward
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        all_hidden_states = outputs.hidden_states

        # 1. Weighted Layer Pooling
        # [batch, seq, hidden]
        sequence_output = self.layer_pooler(all_hidden_states)

        # 2. Hybrid Pooling
        # A. Linear Attention Pooling
        att_feature = self.attention_pooler(sequence_output, attention_mask)

        # B. Global Max Pooling
        # Mask padding tokens to avoid selecting them as max
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(sequence_output.size()).float()
        )
        sequence_output_masked = sequence_output.clone()
        sequence_output_masked[input_mask_expanded == 0] = -1e9
        max_feature = torch.max(sequence_output_masked, dim=1)[0]

        # Concatenate features
        feature = torch.cat([att_feature, max_feature], dim=1)

        # 3. Multi-Sample Dropout & Classification
        logits_list = []
        for i in range(Config.msd_num):
            # Apply dropout branch i
            dropped_feature = self.dropouts[i](feature)
            # Apply classifier
            logits_list.append(self.fc(dropped_feature))

        # Average the logits from all dropout branches
        logits = torch.mean(torch.stack(logits_list), dim=0)

        return logits
