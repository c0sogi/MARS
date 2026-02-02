import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from library.config import Config


class TweetModel(nn.Module):
    """
    TweetModel architecture for Sentiment Extraction.

    Features:
    - Backbone: DeBERTa-v3-base
    - Head: Weighted Layer Pooling (aggregates last N layers)
    - Regularization: Multi-Sample Dropout (ensembles dropout masks)
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # 1. Load Configuration and Backbone
        # We need hidden states for Weighted Layer Pooling
        self.hf_config = AutoConfig.from_pretrained(
            config.model_name, output_hidden_states=True
        )

        self.backbone = AutoModel.from_pretrained(
            config.model_name, config=self.hf_config
        )

        # 2. Weighted Layer Pooling Components
        # Learnable weights for the last N layers
        self.n_layers = config.weighted_layer_pooling_layers
        self.layer_weights = nn.Parameter(torch.ones(self.n_layers))

        # 3. Multi-Sample Dropout Components
        # A list of dropout layers with different rates
        self.dropouts = nn.ModuleList(
            [nn.Dropout(p) for p in config.multi_sample_dropout_rates]
        )

        # 4. Classification Head
        # Projects hidden_size -> 2 (start_logit, end_logit)
        self.fc = nn.Linear(self.hf_config.hidden_size, 2)

        # Initialize weights for the custom head
        self._init_weights(self.fc)

    def _init_weights(self, module):
        """
        Initialize weights for the classification head using the backbone's initializer range.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.hf_config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Shape (batch_size, seq_len)
            attention_mask (torch.Tensor): Shape (batch_size, seq_len)
            token_type_ids (torch.Tensor): Shape (batch_size, seq_len)

        Returns:
            start_logits (torch.Tensor): Shape (batch_size, seq_len)
            end_logits (torch.Tensor): Shape (batch_size, seq_len)
        """
        # 1. Backbone Forward
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # outputs.hidden_states is a tuple of tensors: (embeddings, layer_1, ..., layer_12)
        all_hidden_states = outputs.hidden_states

        # 2. Weighted Layer Pooling
        # Extract the last N layers
        layers_to_pool = all_hidden_states[-self.n_layers :]

        # Stack layers to shape: (batch, seq, hidden, n_layers)
        stacked_layers = torch.stack(layers_to_pool, dim=-1)

        # Compute softmax weights: (n_layers,) -> (1, 1, 1, n_layers) for broadcasting
        weights = torch.softmax(self.layer_weights, dim=0)
        weights = weights.view(1, 1, 1, -1)

        # Compute weighted sum across the layer dimension
        # Result shape: (batch, seq, hidden)
        pooled_output = (stacked_layers * weights).sum(dim=-1)

        # 3. Multi-Sample Dropout & Classification
        # Apply multiple dropout masks and average the logits
        start_logits_list = []
        end_logits_list = []

        for dropout in self.dropouts:
            # Apply dropout
            dropped_output = dropout(pooled_output)

            # Project to logits: (batch, seq, 2)
            logits = self.fc(dropped_output)

            # Split into start and end logits
            start, end = logits.split(1, dim=-1)

            start_logits_list.append(start.squeeze(-1))
            end_logits_list.append(end.squeeze(-1))

        # Average the predictions
        start_logits = torch.stack(start_logits_list).mean(dim=0)
        end_logits = torch.stack(end_logits_list).mean(dim=0)

        return start_logits, end_logits
