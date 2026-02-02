import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from library.config import Config


class WeightedLayerPooling(nn.Module):
    """
    Weighted Layer Pooling strategy.
    Dynamically learns weights to aggregate the last N hidden layers of the model.
    """

    def __init__(self, num_hidden_layers):
        super(WeightedLayerPooling, self).__init__()
        self.num_hidden_layers = num_hidden_layers
        # Initialize with equal weights
        self.layer_weights = nn.Parameter(torch.ones(num_hidden_layers))

    def forward(self, all_hidden_states):
        # Select the last N layers
        # all_hidden_states is a tuple of tensors (batch, seq_len, hidden_size)
        # Stack them to get (num_layers, batch, seq_len, hidden_size)
        selected_layers = torch.stack(all_hidden_states[-self.num_hidden_layers :])

        # Compute softmax weights: (num_layers, 1, 1, 1)
        weights = torch.softmax(self.layer_weights, dim=0).view(-1, 1, 1, 1)

        # Weighted sum
        weighted_embedding = (weights * selected_layers).sum(dim=0)
        return weighted_embedding


class SentimentModel(nn.Module):
    """
    DeBERTa-v3-base with Dual-Head Multi-Task Learning.
    1. Primary Span Head: Weighted Layer Pooling -> 1D Conv -> Linear -> Start/End Logits
    2. Auxiliary Dense Head: Weighted Layer Pooling -> Linear -> Token Classification Logits
    """

    def __init__(self):
        super(SentimentModel, self).__init__()
        self.config = AutoConfig.from_pretrained(Config.model_name)
        self.config.output_hidden_states = True

        # Backbone
        self.backbone = AutoModel.from_pretrained(Config.model_name, config=self.config)

        # Pooling
        self.pooling = WeightedLayerPooling(Config.n_pool_layers)

        # Primary Span Head (CNN-Enhanced)
        # 1D Convolution to capture local boundary context
        # Input: (Batch, Hidden, Seq) -> Output: (Batch, Hidden, Seq)
        self.conv = nn.Conv1d(
            in_channels=self.config.hidden_size,
            out_channels=self.config.hidden_size,
            kernel_size=Config.cnn_kernel_size,
            padding=(Config.cnn_kernel_size - 1) // 2,
        )
        self.relu = nn.ReLU()
        self.span_head = nn.Linear(self.config.hidden_size, 2)

        self.dropout = nn.Dropout(Config.dropout)

        # Initialize custom layers
        self._init_custom_weights()

    def _init_custom_weights(self):
        """Initialize weights for custom layers."""
        for module in [self.conv, self.span_head]:
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Conv1d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        # Pass through backbone
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # Get all hidden states
        hidden_states = outputs.hidden_states

        # Aggregate last N layers
        feature = self.pooling(hidden_states)  # (Batch, Seq, Hidden)
        feature = self.dropout(feature)

        # --- Primary Span Head ---
        # Permute for Conv1d: (Batch, Hidden, Seq)
        cnn_input = feature.permute(0, 2, 1)
        cnn_out = self.conv(cnn_input)
        cnn_out = self.relu(cnn_out)
        # Permute back: (Batch, Seq, Hidden)
        cnn_out = cnn_out.permute(0, 2, 1)

        # Project to Start/End logits
        span_logits = self.span_head(cnn_out)  # (Batch, Seq, 2)
        start_logits, end_logits = span_logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)  # (Batch, Seq)
        end_logits = end_logits.squeeze(-1)  # (Batch, Seq)

        return start_logits, end_logits
