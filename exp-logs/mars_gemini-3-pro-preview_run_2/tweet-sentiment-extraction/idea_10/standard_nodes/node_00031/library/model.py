import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class WeightedLayerPooling(nn.Module):
    """
    Weighted Layer Pooling: Learns a weighted average of the last N hidden states.
    This allows the model to dynamically select the most relevant layers for the task.
    """

    def __init__(self, num_hidden_layers):
        super(WeightedLayerPooling, self).__init__()
        self.num_hidden_layers = num_hidden_layers
        # Initialize weights to be equal (1.0) before softmax
        self.layer_weights = nn.Parameter(
            torch.tensor([1] * num_hidden_layers, dtype=torch.float)
        )

    def forward(self, all_hidden_states):
        # all_hidden_states is a tuple of tensors (batch, seq_len, hidden_size)
        # We take the last 'num_hidden_layers'
        # Stack them to shape: (num_layers, batch, seq_len, hidden_size)
        all_layer_embedding = torch.stack(
            list(all_hidden_states)[-self.num_hidden_layers :], dim=0
        )

        # Calculate softmax weights: (num_layers, 1, 1, 1) for broadcasting
        weight_factor = torch.softmax(self.layer_weights, dim=0).view(-1, 1, 1, 1)

        # Weighted sum: (batch, seq_len, hidden_size)
        weighted_average = (weight_factor * all_layer_embedding).sum(dim=0)
        return weighted_average


class TweetModel(nn.Module):
    """
    Model class for Tweet Sentiment Extraction.
    Backbone: microsoft/deberta-v3-large
    Head: Weighted Layer Pooling -> 1D CNN -> GELU -> Linear
    """

    def __init__(self, config):
        super(TweetModel, self).__init__()
        self.config = config

        # Load Transformer Configuration
        # Ensure output_hidden_states is True to enable WeightedLayerPooling
        transformer_config = AutoConfig.from_pretrained(config.model_name)
        transformer_config.output_hidden_states = True

        # Load Transformer Backbone
        self.backbone = AutoModel.from_pretrained(
            config.model_name, config=transformer_config
        )

        # Weighted Layer Pooling
        self.pooler = WeightedLayerPooling(num_hidden_layers=config.num_pooling_layers)

        # CNN-Enhanced Span Head
        # The CNN refines local boundary features (n-grams)
        # Conv1d expects (Batch, Channels, Length)
        self.cnn = nn.Conv1d(
            in_channels=config.hidden_size,
            out_channels=config.cnn_mid_channels,
            kernel_size=config.cnn_kernel_size,
            padding=config.cnn_padding,
        )
        self.act = nn.GELU()

        # Final Linear Predictor
        # Maps from cnn_mid_channels to 2 (start_logits, end_logits)
        self.fc = nn.Linear(config.cnn_mid_channels, 2)

        # Initialize custom layers
        self._init_weights(self.cnn)
        self._init_weights(self.fc)

    def _init_weights(self, module):
        """
        Initialize weights for the custom head layers.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Conv1d):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        # Backbone Forward
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # Weighted Pooling of Hidden States
        # outputs.hidden_states is a tuple of (Batch, SeqLen, Hidden)
        sequence_output = self.pooler(outputs.hidden_states)  # (Batch, SeqLen, Hidden)

        # CNN Head
        # Permute for Conv1d: (Batch, Hidden, SeqLen)
        x = sequence_output.permute(0, 2, 1)
        x = self.cnn(x)
        x = self.act(x)

        # Permute back: (Batch, SeqLen, MidChannels)
        x = x.permute(0, 2, 1)

        # Linear Predictor
        logits = self.fc(x)  # (Batch, SeqLen, 2)

        # Split into start and end logits
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)  # (Batch, SeqLen)
        end_logits = end_logits.squeeze(-1)  # (Batch, SeqLen)

        return start_logits, end_logits
