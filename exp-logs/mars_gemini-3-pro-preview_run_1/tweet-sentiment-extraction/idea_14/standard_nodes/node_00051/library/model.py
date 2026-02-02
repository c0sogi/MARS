import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config


class WeightedLayerPooling(nn.Module):
    """
    Weighted Layer Pooling strategy.
    Aggregates the hidden states of the last N layers using learnable weights.
    """

    def __init__(self, num_hidden_layers=24, layer_start=20):
        super(WeightedLayerPooling, self).__init__()
        # We focus on the last 4 layers by default as per the design
        self.num_pool_layers = 4
        # Initialize weights to 0, which results in equal weighting after softmax
        self.weights = nn.Parameter(torch.zeros(self.num_pool_layers))

    def forward(self, all_hidden_states):
        # all_hidden_states is a tuple of (embeddings + num_layers) tensors
        # We take the last 4 layers
        states = all_hidden_states[-self.num_pool_layers :]

        # Stack to (num_pool_layers, batch, seq, dim)
        stacked = torch.stack(states, dim=0)

        # Compute softmax weights: (num_pool_layers)
        weights = torch.softmax(self.weights, dim=0)

        # Reshape for broadcasting: (num_pool_layers, 1, 1, 1)
        weights = weights.view(-1, 1, 1, 1)

        # Weighted sum along the layer dimension
        weighted_sum = (weights * stacked).sum(dim=0)

        return weighted_sum


class GatedConvHead(nn.Module):
    """
    Gated Convolutional Head.
    Applies a 1D convolution with a gating mechanism to refine features.
    Architecture:
      - Content Stream: Conv1d
      - Gate Stream: Conv1d -> Sigmoid
      - Output: Content * Gate
      - Dropout -> Linear Projection
    """

    def __init__(self, input_dim, kernel_size=3, dropout=0.1):
        super(GatedConvHead, self).__init__()
        # Padding to maintain sequence length: (kernel_size - 1) // 2
        self.padding = (kernel_size - 1) // 2

        self.conv_content = nn.Conv1d(
            input_dim, input_dim, kernel_size, padding=self.padding
        )
        self.conv_gate = nn.Conv1d(
            input_dim, input_dim, kernel_size, padding=self.padding
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(input_dim, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Input x: (batch, seq_len, input_dim)

        # Transpose for Conv1d: (batch, input_dim, seq_len)
        x = x.transpose(1, 2)

        # Content Stream
        c = self.conv_content(x)

        # Gate Stream
        g = self.sigmoid(self.conv_gate(x))

        # Gated Operation
        h = c * g

        # Transpose back: (batch, seq_len, input_dim)
        h = h.transpose(1, 2)

        # Regularization and Projection
        h = self.dropout(h)
        logits = self.classifier(h)

        return logits


class TweetModel(nn.Module):
    """
    Main Model Class.
    Backbone: DeBERTa-v3-Large
    Head: WeightedLayerPooling + GatedConvHead
    """

    def __init__(self, conf=Config):
        super(TweetModel, self).__init__()
        self.config = AutoConfig.from_pretrained(
            conf.model_name, output_hidden_states=True
        )
        self.backbone = AutoModel.from_pretrained(conf.model_name, config=self.config)

        # Feature Aggregation
        self.pooler = WeightedLayerPooling()

        # Custom Head
        self.head = GatedConvHead(
            input_dim=self.config.hidden_size,
            kernel_size=conf.conv_kernel_size,
            dropout=conf.dropout,
        )

        # Initialize custom modules
        self._init_weights(self.pooler)
        self._init_weights(self.head)

    def _init_weights(self, module):
        """
        Initialize weights for the custom head and pooler.
        Uses the initializer range from the backbone config.
        """
        init_range = getattr(self.config, "initializer_range", 0.02)

        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=init_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Conv1d):
            module.weight.data.normal_(mean=0.0, std=init_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=init_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask):
        # Backbone Forward
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        # Pooling (Aggregate last 4 hidden states)
        feature = self.pooler(outputs.hidden_states)

        # Head Forward
        logits = self.head(feature)

        # Split logits into start and end
        # logits shape: (batch, seq, 2)
        start_logits, end_logits = logits.split(1, dim=-1)

        # Squeeze last dim: (batch, seq)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits
