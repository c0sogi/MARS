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


class SentimentDecoupledDeberta(nn.Module):
    """
    DeBERTa-v3 model with Sentiment-Decoupled Heads.

    Architecture:
    1. DeBERTa-v3 Backbone
    2. Weighted Layer Pooling (Last 4 layers)
    3. Shared 1D Convolutional Layer
    4. Decoupled Linear Heads (Positive vs Negative)
    """

    def __init__(self):
        super(SentimentDecoupledDeberta, self).__init__()
        self.config = AutoConfig.from_pretrained(
            Config.MODEL_NAME, output_hidden_states=True
        )
        self.deberta = AutoModel.from_pretrained(Config.MODEL_NAME, config=self.config)

        self.pooling = WeightedLayerPooling(num_hidden_layers=4)

        # Shared Context Layer: 1D Conv to capture local n-gram context
        self.conv = nn.Conv1d(
            in_channels=self.config.hidden_size,
            out_channels=self.config.hidden_size,
            kernel_size=3,
            padding=1,
        )

        self.dropout = nn.Dropout(Config.DROPOUT)

        # Decoupled Heads: Separate projections for Positive and Negative sentiments
        # Output dim is 2 (start_logit, end_logit)
        self.positive_head = nn.Linear(self.config.hidden_size, 2)
        self.negative_head = nn.Linear(self.config.hidden_size, 2)

        # Initialize custom layers
        self._init_weights(self.conv)
        self._init_weights(self.positive_head)
        self._init_weights(self.negative_head)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, input_ids, attention_mask, sentiment=None):
        """
        Forward pass with sentiment-based routing.

        Args:
            input_ids: (Batch, Seq_Len)
            attention_mask: (Batch, Seq_Len)
            sentiment: List or Tuple of strings representing sentiment for each sample.
        """
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.hidden_states

        # 1. Weighted Pooling
        feature = self.pooling(hidden_states)  # (Batch, Seq, Hidden)

        # 2. Shared Convolution
        # Conv1d expects (Batch, Channels, Seq)
        feature = feature.transpose(1, 2)
        feature = self.conv(feature)
        feature = feature.transpose(1, 2)  # Back to (Batch, Seq, Hidden)

        feature = self.dropout(feature)

        # 3. Decoupled Head Routing
        batch_size, seq_len, _ = feature.shape
        logits = torch.zeros(batch_size, seq_len, 2, device=input_ids.device)

        if sentiment is not None:
            # Identify indices for each sentiment class
            pos_indices = [i for i, s in enumerate(sentiment) if s == "positive"]
            neg_indices = [i for i, s in enumerate(sentiment) if s == "negative"]

            # Route Positive samples
            if pos_indices:
                pos_indices_tensor = torch.tensor(pos_indices, device=input_ids.device)
                pos_features = feature[pos_indices_tensor]
                logits[pos_indices_tensor] = self.positive_head(pos_features)

            # Route Negative samples
            if neg_indices:
                neg_indices_tensor = torch.tensor(neg_indices, device=input_ids.device)
                neg_features = feature[neg_indices_tensor]
                logits[neg_indices_tensor] = self.negative_head(neg_features)

            # Neutral samples (if any) remain 0s, as they are not trained/predicted by this model.

        return logits
