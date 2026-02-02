import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from library.config import Config


class WeightedLayerPooling(nn.Module):
    """
    Computes a learnable weighted average of the last n hidden layers.
    """

    def __init__(self, num_hidden_layers):
        super(WeightedLayerPooling, self).__init__()
        self.num_hidden_layers = num_hidden_layers
        self.weights = nn.Parameter(
            torch.tensor([1] * num_hidden_layers, dtype=torch.float)
        )

    def forward(self, all_hidden_states):
        # Extract the last 'num_hidden_layers' hidden states
        # all_hidden_states is a tuple, we take the last n
        extract_layers = all_hidden_states[-self.num_hidden_layers :]

        # Stack layers: (batch, seq_len, hidden_dim, num_layers)
        stacked = torch.stack(extract_layers, dim=-1)

        # Compute softmax weights
        w = torch.softmax(self.weights, dim=0)

        # Compute weighted sum
        # (batch, seq_len, hidden_dim, num_layers) * (1, 1, 1, num_layers) -> sum over last dim
        weighted_sum = (stacked * w.view(1, 1, 1, -1)).sum(dim=-1)
        return weighted_sum


class CustomDeberta(nn.Module):
    """
    Custom DeBERTa architecture with Weighted Layer Pooling, Dual Pooling,
    and Multi-Sample Dropout.
    """

    def __init__(self, model_name=Config.model_name, pretrained=True):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.config.output_hidden_states = True

        if pretrained:
            self.model = AutoModel.from_pretrained(model_name, config=self.config)
        else:
            self.model = AutoModel.from_config(self.config)

        # Weighted Layer Pooling
        self.pooler = WeightedLayerPooling(Config.n_last_hidden_layers)

        # Dual Pooling (Mean + Max) concatenates features, doubling the dimension
        self.fc_input_dim = self.config.hidden_size * 2

        # Multi-Sample Dropout
        self.dropouts = nn.ModuleList(
            [nn.Dropout(Config.dropout) for _ in range(Config.n_dropout_samples)]
        )

        # Final Classifier
        self.fc = nn.Linear(self.fc_input_dim, Config.num_classes)

        # Initialize Head Weights
        self._init_weights(self.fc)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask, labels=None):
        # Backbone Forward Pass
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        all_hidden_states = outputs.hidden_states

        # 1. Weighted Layer Pooling
        # Returns: (batch, seq_len, hidden_dim)
        sequence_output = self.pooler(all_hidden_states)

        # 2. Dual Pooling (Mean + Max)
        # Create mask for pooling operations
        # Expand mask: (batch, seq_len, 1)
        mask_expanded = (
            attention_mask.unsqueeze(-1).expand(sequence_output.size()).float()
        )

        # Mean Pooling
        sum_embeddings = torch.sum(sequence_output * mask_expanded, 1)
        sum_mask = mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_pooling = sum_embeddings / sum_mask

        # Max Pooling
        # Replace padding with very small value so max ignores them
        sequence_output_masked = sequence_output.clone()
        sequence_output_masked[mask_expanded == 0] = -1e9
        max_pooling, _ = torch.max(sequence_output_masked, 1)

        # Concatenate
        embeddings = torch.cat([mean_pooling, max_pooling], dim=1)

        # 3. Multi-Sample Dropout
        logits_list = []
        for dropout in self.dropouts:
            # Apply dropout then classifier
            logits_list.append(self.fc(dropout(embeddings)))

        # Average the predictions from different dropout masks
        logits = torch.mean(torch.stack(logits_list), dim=0)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return {"loss": loss, "logits": logits}
