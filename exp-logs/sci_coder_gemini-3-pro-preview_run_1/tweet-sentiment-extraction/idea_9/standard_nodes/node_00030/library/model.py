import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from library.config import Config


class WeightedLayerPooling(nn.Module):
    """
    Aggregates the last 'num_hidden_layers' layers of the transformer
    using learnable weights.
    """

    def __init__(self, num_hidden_layers=4, layer_start=4):
        super(WeightedLayerPooling, self).__init__()
        self.num_hidden_layers = num_hidden_layers
        self.layer_weights = nn.Parameter(
            torch.tensor([1] * num_hidden_layers, dtype=torch.float)
        )

    def forward(self, all_hidden_states):
        # all_hidden_states is a tuple of tensors (batch, seq_len, hidden_dim)
        # Select the last 'num_hidden_layers'
        selected_layers = all_hidden_states[-self.num_hidden_layers :]

        # Stack to (batch, num_layers, seq_len, hidden_dim)
        all_layer_embedding = torch.stack(selected_layers, dim=1)

        # Compute softmax weights
        weight_factor = F.softmax(self.layer_weights, dim=0).view(1, -1, 1, 1)

        # Weighted sum
        weighted_embedding = (weight_factor * all_layer_embedding).sum(dim=1)
        return weighted_embedding


class TweetModel(nn.Module):
    """
    DeBERTa-v3 model with a Shared Head for Implicit Conditioning.
    Cite solution_lesson_node_00029: Avoid explicit branching; use shared head.
    Cite solution_lesson_node_00011: Use CNN-enhanced head.
    """

    def __init__(self):
        super(TweetModel, self).__init__()
        self.config = AutoConfig.from_pretrained(
            Config.MODEL_NAME, output_hidden_states=True
        )
        self.deberta = AutoModel.from_pretrained(Config.MODEL_NAME, config=self.config)

        self.pooling = WeightedLayerPooling(num_hidden_layers=4)

        # 1D Conv to capture local n-gram context
        self.conv = nn.Conv1d(
            in_channels=self.config.hidden_size,
            out_channels=self.config.hidden_size,
            kernel_size=3,
            padding=1,
        )

        self.dropout = nn.Dropout(Config.DROPOUT)

        # Single shared projection head
        self.qa_outputs = nn.Linear(self.config.hidden_size, 2)

        self._init_weights(self.conv)
        self._init_weights(self.qa_outputs)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, input_ids, attention_mask):
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.hidden_states

        # Weighted Pooling
        feature = self.pooling(hidden_states)

        # Convolution
        feature = feature.transpose(1, 2)
        feature = self.conv(feature)
        feature = feature.transpose(1, 2)

        feature = self.dropout(feature)

        # Shared Head Projection
        logits = self.qa_outputs(feature)

        return logits
